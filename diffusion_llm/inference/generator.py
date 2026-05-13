"""
AAM Diffusion LLM — Inference Generator

Generates natural language narratives from graph conditioning
using the trained diffusion model.

The generation process:
1. Encode graph conditioning (evidence, anomalies, reasoning)
2. Start from pure noise in the latent space
3. Iteratively denoise for N steps
4. Convert denoised embeddings to token IDs
5. Detokenize to natural language text

Analogi: Seperti Jin Soun akhirnya "berbicara" — dari
pikiran yang kabur (noise) menjadi kata-kata yang jelas
(denoised narrative). Setiap langkah denoising = satu
langkah lebih dekat ke koherensi.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from diffusion_llm.config.model_config import AamDiffusionConfig, InferenceConfig
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from a generation call.

    Contains the generated narrative plus metadata about
    how it was generated, for traceability.
    """
    narrative: str
    """Generated narrative text."""

    token_ids: list[int] = field(default_factory=list)
    """Generated token IDs."""

    n_diffusion_steps: int = 0
    """Number of denoising steps used."""

    generation_time_s: float = 0.0
    """Wall-clock generation time."""

    model_name: str = ""
    """Name of the model used."""

    evidence_used: list[str] = field(default_factory=list)
    """Evidence nodes that were provided as conditioning."""

    confidence: float = 0.0
    """Overall confidence of the generation."""

    language: str = "id"
    """Output language."""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "narrative": self.narrative,
            "n_diffusion_steps": self.n_diffusion_steps,
            "generation_time_s": round(self.generation_time_s, 3),
            "model_name": self.model_name,
            "evidence_used": self.evidence_used,
            "confidence": round(self.confidence, 3),
            "language": self.language,
        }


class AamGenerator:
    """Generate narratives from graph conditioning using the trained model.

    This is the main inference interface. It takes graph-structured
    data (from the RSVS Knowledge Graph) and produces natural
    language narratives through the diffusion denoising process.

    Usage:
        # Load model and tokenizer
        config = AamDiffusionConfig.from_json("config.json")
        model = AamDiffusionModel.load("best.pt")
        tokenizer = AamTokenizer.load("tokenizer.json")

        # Create generator
        generator = AamGenerator(model, tokenizer, config)

        # Generate narrative
        result = generator.generate(
            trigger="Siapa yang mencuri Snow Plum Pill?",
            evidence_nodes=["hefei", "diancang", "ju_jangmok"],
            anomalies=["no external pill consumption"],
            reasoning_steps=["Diancang pair was in Hefei before theft"],
        )
        print(result.narrative)

    Args:
        model: Trained AamDiffusionModel.
        tokenizer: Trained AamTokenizer.
        config: AamDiffusionConfig with inference settings.
    """

    def __init__(
        self,
        model: AamDiffusionModel,
        tokenizer: AamTokenizer,
        config: AamDiffusionConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.inference_config = config.inference

        # Device
        self.device = next(model.parameters()).device

        # Set model to eval mode
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        trigger: str = "",
        evidence_nodes: Optional[list[str]] = None,
        compositions: Optional[list[str]] = None,
        confidence_map: Optional[dict[str, float]] = None,
        anomalies: Optional[list[str]] = None,
        reasoning_steps: Optional[list[str]] = None,
        source_trust: float = 1.0,
        n_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        language: Optional[str] = None,
        max_sentences: Optional[int] = None,
    ) -> GenerationResult:
        """Generate a narrative from graph conditioning.

        This is the main generation method. It:
        1. Tokenizes the graph conditioning data
        2. Encodes it through the graph encoder
        3. Starts from noise and iteratively denoises
        4. Converts the result to text

        Args:
            trigger: The trigger question or topic.
            evidence_nodes: Evidence node descriptions.
            compositions: Composition descriptions.
            confidence_map: Node confidence scores.
            anomalies: Anomaly descriptions.
            reasoning_steps: Reasoning step descriptions.
            source_trust: Source trust score.
            n_steps: Override number of denoising steps.
            temperature: Override sampling temperature.
            language: Override output language.
            max_sentences: Maximum sentences in output.

        Returns:
            GenerationResult with the narrative and metadata.
        """
        start_time = time.time()

        # Use config defaults if not overridden
        n_steps = n_steps or self.inference_config.n_steps
        temperature = temperature or self.inference_config.temperature
        language = language or self.inference_config.language
        max_sentences = max_sentences or self.inference_config.max_output_sentences

        # --- Step 1: Tokenize graph conditioning ---
        evidence_ids_tensor = None
        evidence_conf_tensor = None
        anomaly_ids_tensor = None
        anomaly_conf_tensor = None
        reasoning_ids_tensor = None
        reasoning_conf_tensor = None

        if evidence_nodes:
            evidence_ids_list = []
            evidence_conf_list = []
            for node in evidence_nodes[:self.config.graph_encoder.max_evidence_nodes]:
                ids = self.tokenizer.encode(node, add_special=False)
                ids = self.tokenizer.pad_sequence(ids, 32)
                evidence_ids_list.append(ids)
                conf = (confidence_map or {}).get(node, 0.7)
                evidence_conf_list.append(conf)

            while len(evidence_ids_list) < self.config.graph_encoder.max_evidence_nodes:
                evidence_ids_list.append([0] * 32)
                evidence_conf_list.append(0.0)

            evidence_ids_tensor = torch.tensor(
                [evidence_ids_list], dtype=torch.long, device=self.device
            )
            evidence_conf_tensor = torch.tensor(
                [evidence_conf_list], dtype=torch.float32, device=self.device
            )

        if anomalies:
            anomaly_ids_list = []
            for anom in anomalies[:self.config.graph_encoder.max_anomalies]:
                ids = self.tokenizer.encode(anom, add_special=False)
                ids = self.tokenizer.pad_sequence(ids, 32)
                anomaly_ids_list.append(ids)

            while len(anomaly_ids_list) < self.config.graph_encoder.max_anomalies:
                anomaly_ids_list.append([0] * 32)

            anomaly_ids_tensor = torch.tensor(
                [anomaly_ids_list], dtype=torch.long, device=self.device
            )
            anomaly_conf_tensor = torch.full(
                (1, self.config.graph_encoder.max_anomalies),
                0.6, dtype=torch.float32, device=self.device,
            )

        if reasoning_steps:
            reasoning_ids_list = []
            for step in reasoning_steps[:self.config.graph_encoder.max_reasoning_steps]:
                ids = self.tokenizer.encode(step, add_special=False)
                ids = self.tokenizer.pad_sequence(ids, 32)
                reasoning_ids_list.append(ids)

            while len(reasoning_ids_list) < self.config.graph_encoder.max_reasoning_steps:
                reasoning_ids_list.append([0] * 32)

            reasoning_ids_tensor = torch.tensor(
                [reasoning_ids_list], dtype=torch.long, device=self.device
            )
            reasoning_conf_tensor = torch.full(
                (1, self.config.graph_encoder.max_reasoning_steps),
                0.7, dtype=torch.float32, device=self.device,
            )

        source_trust_tensor = torch.tensor(
            [source_trust], dtype=torch.float32, device=self.device
        )

        # --- Step 2: Encode graph conditioning ---
        graph_cond = self.model.graph_encoder(
            evidence_ids=evidence_ids_tensor,
            evidence_confidence=evidence_conf_tensor,
            anomaly_ids=anomaly_ids_tensor,
            anomaly_confidence=anomaly_conf_tensor,
            reasoning_ids=reasoning_ids_tensor,
            reasoning_confidence=reasoning_conf_tensor,
            source_trust=source_trust_tensor,
        )

        # --- Step 3: Generate via diffusion denoising ---
        shape = (
            1,
            self.config.model.max_seq_len,
            self.config.model.d_model,
        )

        denoised = self.model.sample(
            graph_cond=graph_cond,
            n_steps=n_steps,
            method=self.config.diffusion.sampling_method,
            shape=shape,
            device=self.device,
        )

        # --- Step 4: Convert to tokens ---
        token_ids = self.model.embeddings_to_tokens(
            denoised, temperature=temperature,
            top_k=self.inference_config.top_k,
        )

        # --- Step 5: Detokenize ---
        token_list = token_ids[0].cpu().tolist()
        narrative = self.tokenizer.decode(token_list, skip_special=True)

        # Truncate to max sentences
        if max_sentences:
            sentences = self.tokenizer._split_sentences(narrative)
            if len(sentences) > max_sentences:
                narrative = ". ".join(sentences[:max_sentences]) + "."

        generation_time = time.time() - start_time

        # Compute average confidence
        avg_confidence = source_trust
        if confidence_map:
            avg_confidence = sum(confidence_map.values()) / len(confidence_map)

        return GenerationResult(
            narrative=narrative,
            token_ids=token_list,
            n_diffusion_steps=n_steps,
            generation_time_s=generation_time,
            model_name=self.config.model_name,
            evidence_used=evidence_nodes or [],
            confidence=avg_confidence,
            language=language,
        )

    def generate_batch(
        self,
        triggers: list[str],
        evidence_nodes_list: Optional[list[list[str]]] = None,
        anomalies_list: Optional[list[list[str]]] = None,
        **kwargs,
    ) -> list[GenerationResult]:
        """Generate narratives for multiple triggers.

        Args:
            triggers: List of trigger questions.
            evidence_nodes_list: List of evidence node lists.
            anomalies_list: List of anomaly lists.
            **kwargs: Additional arguments passed to generate().

        Returns:
            List of GenerationResult objects.
        """
        results = []
        for i, trigger in enumerate(triggers):
            evidence = evidence_nodes_list[i] if evidence_nodes_list else None
            anomalies = anomalies_list[i] if anomalies_list else None
            result = self.generate(
                trigger=trigger,
                evidence_nodes=evidence,
                anomalies=anomalies,
                **kwargs,
            )
            results.append(result)
        return results
