"""
AAM Diffusion LLM — Dataset

Dataset class for Graph→Narrative training pairs.

Each training example consists of:
    - Graph conditioning: evidence nodes, compositions, confidence,
      anomalies, reasoning chains, temporal context
    - Target narrative: natural language text that represents
      the graph data in sentence form

The dataset handles:
    - Loading from JSONL files
    - Tokenization of both graph data and narratives
    - Padding and batching
    - Data augmentation (sentence shuffling, noise injection)

Analogi: Seperti Jin Soun berlatih mengungkapkan kesimpulan —
dia diberi "kasus" (graph data) dan "jawaban yang benar"
(narrative target), lalu berlatih sampai bisa menyusun
kalimat yang tepat dari graph.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GraphNarrativeExample:
    """A single training example: graph conditioning + target narrative.

    This represents the "input" and "expected output" for one
    training step of the diffusion model.
    """
    # Target narrative (what the model should generate)
    narrative: str = ""

    # Graph conditioning inputs
    trigger: str = ""
    evidence_nodes: list[str] = field(default_factory=list)
    compositions: list[str] = field(default_factory=list)
    confidence_map: dict[str, float] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    source_trust: float = 1.0
    temporal_context: list[str] = field(default_factory=list)

    # Metadata
    language: str = "id"
    source: str = "synthetic"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "narrative": self.narrative,
            "trigger": self.trigger,
            "evidence_nodes": self.evidence_nodes,
            "compositions": self.compositions,
            "confidence_map": self.confidence_map,
            "anomalies": self.anomalies,
            "reasoning_steps": self.reasoning_steps,
            "source_trust": self.source_trust,
            "temporal_context": self.temporal_context,
            "language": self.language,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GraphNarrativeExample:
        """Deserialize from dictionary."""
        return cls(
            narrative=data.get("narrative", ""),
            trigger=data.get("trigger", ""),
            evidence_nodes=data.get("evidence_nodes", []),
            compositions=data.get("compositions", []),
            confidence_map=data.get("confidence_map", {}),
            anomalies=data.get("anomalies", []),
            reasoning_steps=data.get("reasoning_steps", []),
            source_trust=data.get("source_trust", 1.0),
            temporal_context=data.get("temporal_context", []),
            language=data.get("language", "id"),
            source=data.get("source", "synthetic"),
        )


@dataclass
class BatchOutput:
    """Output from a single batch.

    All tensors are already padded to uniform length.
    """
    token_ids: torch.Tensor
    """Target narrative token IDs, shape (batch, seq_len)."""

    evidence_ids: Optional[torch.Tensor] = None
    """Evidence node token IDs, shape (batch, n_evidence, ev_seq_len)."""

    evidence_confidence: Optional[torch.Tensor] = None
    """Evidence confidence, shape (batch, n_evidence)."""

    anomaly_ids: Optional[torch.Tensor] = None
    """Anomaly token IDs, shape (batch, n_anomalies, an_seq_len)."""

    anomaly_confidence: Optional[torch.Tensor] = None
    """Anomaly confidence, shape (batch, n_anomalies)."""

    reasoning_ids: Optional[torch.Tensor] = None
    """Reasoning step token IDs, shape (batch, n_steps, r_seq_len)."""

    reasoning_confidence: Optional[torch.Tensor] = None
    """Reasoning confidence, shape (batch, n_steps)."""

    source_trust: Optional[torch.Tensor] = None
    """Source trust scores, shape (batch,)."""


class GraphNarrativeDataset(Dataset):
    """Dataset for Graph→Narrative training pairs.

    Loads training examples from JSONL files and provides
    tokenized, padded batches for training.

    Args:
        data_path: Path to JSONL file with training data.
        tokenizer: AamTokenizer instance for encoding.
        max_seq_len: Maximum sequence length for narratives.
        max_evidence: Maximum number of evidence nodes.
        max_anomalies: Maximum number of anomalies.
        max_reasoning: Maximum number of reasoning steps.
        augment: Whether to apply data augmentation.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: AamTokenizer,
        max_seq_len: int = 512,
        max_evidence: int = 50,
        max_anomalies: int = 10,
        max_reasoning: int = 15,
        augment: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_evidence = max_evidence
        self.max_anomalies = max_anomalies
        self.max_reasoning = max_reasoning
        self.augment = augment

        # Load data
        self.examples: list[GraphNarrativeExample] = []
        self._load_data()

        logger.info(
            "GraphNarrativeDataset: %d examples loaded from %s",
            len(self.examples),
            self.data_path,
        )

    def _load_data(self) -> None:
        """Load examples from JSONL file."""
        if not self.data_path.exists():
            logger.warning("Data file not found: %s", self.data_path)
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    example = GraphNarrativeExample.from_dict(data)
                    if example.narrative:  # Skip empty narratives
                        self.examples.append(example)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON at line %d", line_num)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training example.

        Returns:
            Dictionary with tokenized inputs.
        """
        example = self.examples[idx]

        # Data augmentation
        if self.augment:
            example = self._augment(example)

        # Tokenize narrative (target)
        narrative_ids = self.tokenizer.encode(example.narrative, add_special=True)
        narrative_ids = self.tokenizer.pad_sequence(narrative_ids, self.max_seq_len)
        narrative_tensor = torch.tensor(narrative_ids, dtype=torch.long)

        # Tokenize evidence nodes
        evidence_data = self._tokenize_node_list(
            example.evidence_nodes, max_nodes=self.max_evidence
        )

        # Tokenize anomalies
        anomaly_data = self._tokenize_node_list(
            example.anomalies, max_nodes=self.max_anomalies
        )

        # Tokenize reasoning steps
        reasoning_data = self._tokenize_node_list(
            example.reasoning_steps, max_nodes=self.max_reasoning
        )

        # Source trust
        source_trust = torch.tensor(example.source_trust, dtype=torch.float32)

        # Evidence confidence
        conf_values = list(example.confidence_map.values())[:self.max_evidence]
        if conf_values:
            evidence_conf = torch.tensor(conf_values, dtype=torch.float32)
            evidence_conf = torch.nn.functional.pad(
                evidence_conf, (0, self.max_evidence - len(conf_values))
            )
        else:
            evidence_conf = torch.zeros(self.max_evidence, dtype=torch.float32)

        # Anomaly confidence (default 0.6 for detected anomalies)
        anomaly_conf = torch.full(
            (self.max_anomalies,), 0.6, dtype=torch.float32
        )

        # Reasoning confidence (default 0.7)
        reasoning_conf = torch.full(
            (self.max_reasoning,), 0.7, dtype=torch.float32
        )

        return {
            "token_ids": narrative_tensor,
            "evidence_ids": evidence_data["ids"],
            "evidence_confidence": evidence_conf,
            "anomaly_ids": anomaly_data["ids"],
            "anomaly_confidence": anomaly_conf,
            "reasoning_ids": reasoning_data["ids"],
            "reasoning_confidence": reasoning_conf,
            "source_trust": source_trust,
        }

    def _tokenize_node_list(
        self,
        nodes: list[str],
        max_nodes: int,
        max_node_len: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a list of node descriptions.

        Args:
            nodes: List of node text descriptions.
            max_nodes: Maximum number of nodes to encode.
            max_node_len: Maximum token length per node.

        Returns:
            Dictionary with padded token IDs tensor.
        """
        if not nodes:
            return {
                "ids": torch.zeros(max_nodes, max_node_len, dtype=torch.long),
            }

        # Limit to max_nodes
        nodes = nodes[:max_nodes]

        all_ids = []
        for node in nodes:
            ids = self.tokenizer.encode(node, add_special=False)
            ids = self.tokenizer.pad_sequence(ids, max_node_len)
            all_ids.append(ids)

        # Pad to max_nodes
        while len(all_ids) < max_nodes:
            all_ids.append([0] * max_node_len)

        return {
            "ids": torch.tensor(all_ids, dtype=torch.long),
        }

    def _augment(self, example: GraphNarrativeExample) -> GraphNarrativeExample:
        """Apply data augmentation.

        Augmentation strategies:
        1. Random sentence shuffling within the narrative
        2. Random evidence node dropping (simulate incomplete data)
        3. Random confidence perturbation

        Args:
            example: Original training example.

        Returns:
            Augmented example.
        """
        import copy
        augmented = copy.deepcopy(example)

        # 1. Sentence shuffling (with 20% probability)
        if random.random() < 0.2:
            sentences = self.tokenizer._split_sentences(augmented.narrative)
            if len(sentences) > 2:
                # Keep first sentence, shuffle the rest
                first = sentences[0]
                rest = sentences[1:]
                random.shuffle(rest)
                augmented.narrative = first + " " + " ".join(rest)

        # 2. Evidence dropping (with 10% probability per node)
        if augmented.evidence_nodes:
            augmented.evidence_nodes = [
                node for node in augmented.evidence_nodes
                if random.random() > 0.1
            ]

        # 3. Confidence perturbation
        if augmented.confidence_map:
            perturbed = {}
            for k, v in augmented.confidence_map.items():
                noise = random.gauss(0, 0.05)
                perturbed[k] = max(0.0, min(1.0, v + noise))
            augmented.confidence_map = perturbed

        return augmented


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom collate function for DataLoader.

    Handles variable-length graph conditioning by padding
    all tensors in the batch to the same size.

    Args:
        batch: List of example dictionaries.

    Returns:
        Batched dictionary of tensors.
    """
    result = {}

    # Stack all tensors
    for key in batch[0]:
        tensors = [item[key] for item in batch]
        if tensors[0].dim() == 0:
            result[key] = torch.stack(tensors)
        elif tensors[0].dim() == 1:
            result[key] = torch.stack(tensors)
        elif tensors[0].dim() == 2:
            result[key] = torch.stack(tensors)
        else:
            result[key] = torch.stack(tensors)

    return result
