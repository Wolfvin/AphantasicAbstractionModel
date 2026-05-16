"""
AAM Diffusion LLM — Inference Generator (v2.0)

Generates natural language narratives from graph conditioning
using the trained diffusion model.

v2.0 Upgrades:
    - ThinkingToggle for adaptive inference (thinking vs non-thinking)
    - Anchored decoding method (2-3 steps instead of 50)
    - Flow matching method (velocity-based 2-3 step sampling)
    - MCTS integration for complex reasoning tasks
    - DualMemorySystem for long narrative generation
    - Full backward compatibility with v1.0 generation

The generation process (v2.0 Anchored):
1. Encode graph conditioning (evidence, anomalies, reasoning)
2. [Optional] ThinkingToggle assesses complexity
3. [Optional] MCTS explores narrative arrangements for complex inputs
4. Generate via anchored decoding (2-3 refinement steps)
5. Convert denoised embeddings to token IDs
6. Detokenize to natural language text

The generation process (Legacy DDPM/DDIM):
1. Encode graph conditioning
2. Start from pure noise in the latent space
3. Iteratively denoise for N steps
4. Convert denoised embeddings to token IDs
5. Detokenize to natural language text

Analogi: Seperti Jin Soun akhirnya "berbicara" — dari
pikiran yang kabur (noise) menjadi kata-kata yang jelas
(denoised narrative). Di v2.0, Jin Soun sekarang bisa
memilih: berbicara cepat untuk hal sederhana (non-thinking),
atau berpikir dalam untuk masalah rumit (thinking + MCTS).
"""

from __future__ import annotations

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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

    # v2.0 metadata
    sampling_method: str = "ddim"
    """Sampling method used ('anchored', 'flow_matching', 'ddpm', 'ddim')."""

    thinking_mode: str = ""
    """ThinkingToggle mode: 'thinking', 'non_thinking', or '' if disabled."""

    complexity_score: float = 0.0
    """Complexity score from ThinkingToggle (0.0 if disabled)."""

    mcts_used: bool = False
    """Whether MCTS reasoning was used."""

    memory_stats: Dict[str, object] = field(default_factory=dict)
    """DualMemory statistics at generation time."""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "narrative": self.narrative,
            "n_diffusion_steps": self.n_diffusion_steps,
            "generation_time_s": round(self.generation_time_s, 3),
            "model_name": self.model_name,
            "evidence_used": self.evidence_used,
            "confidence": round(self.confidence, 3),
            "language": self.language,
            "sampling_method": self.sampling_method,
        }
        if self.thinking_mode:
            result["thinking_mode"] = self.thinking_mode
            result["complexity_score"] = round(self.complexity_score, 3)
        if self.mcts_used:
            result["mcts_used"] = True
        if self.memory_stats:
            result["memory_stats"] = self.memory_stats
        return result


class AamGenerator:
    """Generate narratives from graph conditioning using the trained model (v2.0).

    This is the main inference interface. It takes graph-structured
    data (from the RSVS Knowledge Graph) and produces natural
    language narratives through the diffusion denoising process.

    v2.0 features:
    - Adaptive compute via ThinkingToggle
    - Fast anchored decoding (2-3 steps)
    - Flow matching decoding
    - MCTS for complex reasoning
    - Dual memory for long narratives

    Usage:
        # Load model and tokenizer
        config = AamDiffusionConfig.from_json("config.json")
        model = AamDiffusionModel.load("best.pt")
        tokenizer = AamTokenizer.load("tokenizer.json")

        # Create generator
        generator = AamGenerator(model, tokenizer, config)

        # Generate narrative (v2.0 anchored decoding)
        result = generator.generate(
            trigger="Siapa yang mencuri Snow Plum Pill?",
            evidence_nodes=["hefei", "diancang", "ju_jangmok"],
            anomalies=["no external pill consumption"],
            reasoning_steps=["Diancang pair was in Hefei before theft"],
            method="anchored",
        )
        print(result.narrative)

        # Generate narrative (legacy DDIM)
        result = generator.generate(
            trigger="Summary of events",
            evidence_nodes=["event_a", "event_b"],
            method="ddim",
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

        # Feature detection
        self._has_anchored_decoder = hasattr(model, "output_head")
        self._has_thinking_toggle = hasattr(model, "thinking_toggle")
        self._has_flow_matching = hasattr(model, "flow_matching_decoder")
        self._has_mcts = hasattr(model, "mcts_reasoner")
        self._has_dual_memory = hasattr(model, "dual_memory")
        self._has_evoformer = hasattr(model, "evoformer")

        logger.info(
            "AamGenerator v2.0 initialized. Features: anchored=%s, thinking=%s, "
            "flow=%s, mcts=%s, memory=%s, evoformer=%s",
            self._has_anchored_decoder,
            self._has_thinking_toggle,
            self._has_flow_matching,
            self._has_mcts,
            self._has_dual_memory,
            self._has_evoformer,
        )

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
        method: Optional[str] = None,
        use_mcts: Optional[bool] = None,
        force_thinking_mode: Optional[str] = None,
    ) -> GenerationResult:
        """Generate a narrative from graph conditioning.

        This is the main generation method. It:
        1. Tokenizes the graph conditioning data
        2. Encodes it through the graph encoder
        3. [v2.0] Optionally assesses thinking complexity
        4. [v2.0] Optionally runs MCTS for complex reasoning
        5. Generates via the selected sampling method
        6. Converts the result to text

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
            method: Sampling method — 'anchored', 'flow_matching',
                'ddpm', 'ddim', or None (uses config default).
            use_mcts: Override whether to use MCTS. None = auto-decide
                based on ThinkingToggle assessment.
            force_thinking_mode: Force thinking mode ('thinking' or
                'non_thinking'). None = auto-decide.

        Returns:
            GenerationResult with the narrative and metadata.
        """
        start_time = time.time()

        # Use config defaults if not overridden
        n_steps = n_steps or self.inference_config.n_steps
        temperature = temperature or self.inference_config.temperature
        language = language or self.inference_config.language
        max_sentences = max_sentences or self.inference_config.max_output_sentences

        # Determine sampling method
        if method is None:
            # Default to anchored if available, else use config
            if self._has_anchored_decoder:
                method = "anchored"
            else:
                method = self.config.diffusion.sampling_method

        # Validate method availability
        if method == "anchored" and not self._has_anchored_decoder:
            logger.warning(
                "Anchored decoding requested but ContinuousOutputHead not "
                "available. Falling back to '%s'.",
                self.config.diffusion.sampling_method,
            )
            method = self.config.diffusion.sampling_method

        if method == "flow_matching" and not self._has_flow_matching:
            logger.warning(
                "Flow matching requested but FlowMatchingDecoder not "
                "available. Falling back to '%s'.",
                self.config.diffusion.sampling_method,
            )
            method = self.config.diffusion.sampling_method

        # --- Step 1: Tokenize graph conditioning ---
        (
            evidence_ids_tensor,
            evidence_conf_tensor,
            anomaly_ids_tensor,
            anomaly_conf_tensor,
            reasoning_ids_tensor,
            reasoning_conf_tensor,
            composition_ids_tensor,
            composition_conf_tensor,
        ) = self._tokenize_graph_conditioning(
            evidence_nodes=evidence_nodes,
            compositions=compositions,
            confidence_map=confidence_map,
            anomalies=anomalies,
            reasoning_steps=reasoning_steps,
            source_trust=source_trust,
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
            composition_ids=composition_ids_tensor,
            composition_confidence=composition_conf_tensor,
            source_trust=source_trust_tensor,
        )

        # --- Step 3: ThinkingToggle assessment ---
        thinking_mode_str = ""
        complexity_score = 0.0
        assessment = None

        if self._has_thinking_toggle:
            assessment = self._assess_complexity(
                graph_cond, force_thinking_mode=force_thinking_mode
            )
            if assessment is not None:
                thinking_mode_str = assessment.mode.value
                complexity_score = (
                    assessment.complexity_score.mean().item()
                    if assessment.complexity_score.numel() > 0
                    else 0.0
                )

                # Adaptive step count based on thinking assessment
                if method == "anchored":
                    depth_mult = assessment.depth_multiplier.mean().item()
                    n_steps = max(2, min(5, int(3 * depth_mult)))
                elif method in ("ddpm", "ddim"):
                    depth_mult = assessment.depth_multiplier.mean().item()
                    n_steps = max(
                        10,
                        int(self.inference_config.n_steps * depth_mult),
                    )

                logger.debug(
                    "ThinkingToggle: mode=%s, complexity=%.3f, "
                    "depth_mult=%.2f, n_steps=%d",
                    thinking_mode_str,
                    complexity_score,
                    assessment.depth_multiplier.mean().item(),
                    n_steps,
                )

        # --- Step 4: MCTS reasoning (for complex inputs) ---
        mcts_used = False
        mcts_info: Dict[str, Any] = {}

        should_use_mcts = self._should_use_mcts(
            use_mcts=use_mcts,
            assessment=assessment,
            method=method,
        )

        if should_use_mcts:
            mcts_result = self._run_mcts_reasoning(graph_cond)
            if mcts_result is not None:
                mcts_used = True
                mcts_info = mcts_result

        # --- Step 5: Generate via diffusion denoising ---
        shape = (
            1,
            self.config.model.max_seq_len,
            self.config.model.d_model,
        )

        denoised = self.model.sample(
            graph_cond=graph_cond,
            n_steps=n_steps,
            method=method,
            shape=shape,
            device=self.device,
            temperature=temperature,
        )

        # --- Step 6: Convert to tokens ---
        # Extract graph context for anchored decoder
        graph_values = graph_cond.get("values")
        graph_context = None
        if graph_values is not None:
            graph_context = graph_values.mean(dim=1)

        token_ids = self.model.embeddings_to_tokens(
            denoised,
            temperature=temperature,
            top_k=self.inference_config.top_k,
            graph_context=graph_context,
        )

        # --- Step 7: Detokenize ---
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

        # Collect memory stats
        mem_stats = self.model.memory_stats() if self._has_dual_memory else {}

        # Consolidate memory for future generations
        if self._has_dual_memory:
            self.model.memory_consolidate()

        return GenerationResult(
            narrative=narrative,
            token_ids=token_list,
            n_diffusion_steps=n_steps,
            generation_time_s=generation_time,
            model_name=self.config.model_name,
            evidence_used=evidence_nodes or [],
            confidence=avg_confidence,
            language=language,
            sampling_method=method,
            thinking_mode=thinking_mode_str,
            complexity_score=complexity_score,
            mcts_used=mcts_used,
            memory_stats=mem_stats,
        )

    # ================================================================
    # Internal helpers
    # ================================================================

    def _tokenize_graph_conditioning(
        self,
        evidence_nodes: Optional[list[str]] = None,
        compositions: Optional[list[str]] = None,
        confidence_map: Optional[dict[str, float]] = None,
        anomalies: Optional[list[str]] = None,
        reasoning_steps: Optional[list[str]] = None,
        source_trust: float = 1.0,
    ) -> tuple:
        """Tokenize all graph conditioning data into tensors.

        Returns:
            Tuple of (evidence_ids, evidence_conf, anomaly_ids,
            anomaly_conf, reasoning_ids, reasoning_conf,
            composition_ids, composition_conf) tensors.
        """
        evidence_ids_tensor = None
        evidence_conf_tensor = None
        anomaly_ids_tensor = None
        anomaly_conf_tensor = None
        reasoning_ids_tensor = None
        reasoning_conf_tensor = None
        composition_ids_tensor = None
        composition_conf_tensor = None

        max_evidence = self.config.graph_encoder.max_evidence_nodes
        max_anomalies = self.config.graph_encoder.max_anomalies
        max_reasoning = self.config.graph_encoder.max_reasoning_steps
        max_compositions = self.config.graph_encoder.max_compositions
        node_len = 32

        # Evidence nodes
        if evidence_nodes:
            evidence_ids_list = []
            evidence_conf_list = []
            for node in evidence_nodes[:max_evidence]:
                ids = self.tokenizer.encode(node, add_special=False)
                ids = self.tokenizer.pad_sequence(ids, node_len)
                evidence_ids_list.append(ids)
                conf = (confidence_map or {}).get(node, 0.7)
                evidence_conf_list.append(conf)

            while len(evidence_ids_list) < max_evidence:
                evidence_ids_list.append([0] * node_len)
                evidence_conf_list.append(0.0)

            evidence_ids_tensor = torch.tensor(
                [evidence_ids_list], dtype=torch.long, device=self.device
            )
            evidence_conf_tensor = torch.tensor(
                [evidence_conf_list], dtype=torch.float32, device=self.device
            )

        # Compositions
        if compositions:
            composition_ids_list = []
            composition_conf_list = []
            for comp in compositions[:max_compositions]:
                ids = self.tokenizer.encode(comp, add_special=False)
                ids = self.tokenizer.pad_sequence(ids, node_len)
                composition_ids_list.append(ids)
                composition_conf_list.append(0.8)

            while len(composition_ids_list) < max_compositions:
                composition_ids_list.append([0] * node_len)
                composition_conf_list.append(0.0)

            composition_ids_tensor = torch.tensor(
                [composition_ids_list], dtype=torch.long, device=self.device
            )
            composition_conf_tensor = torch.tensor(
                [composition_conf_list], dtype=torch.float32, device=self.device
            )

        # Anomalies
        if anomalies:
            anomaly_ids_list = []
            for anom in anomalies[:max_anomalies]:
                ids = self.tokenizer.encode(anom, add_special=False)
                ids = self.tokenizer.pad_sequence(ids, node_len)
                anomaly_ids_list.append(ids)

            while len(anomaly_ids_list) < max_anomalies:
                anomaly_ids_list.append([0] * node_len)

            anomaly_ids_tensor = torch.tensor(
                [anomaly_ids_list], dtype=torch.long, device=self.device
            )
            anomaly_conf_tensor = torch.full(
                (1, max_anomalies),
                0.6, dtype=torch.float32, device=self.device,
            )

        # Reasoning steps
        if reasoning_steps:
            reasoning_ids_list = []
            for step in reasoning_steps[:max_reasoning]:
                ids = self.tokenizer.encode(step, add_special=False)
                ids = self.tokenizer.pad_sequence(ids, node_len)
                reasoning_ids_list.append(ids)

            while len(reasoning_ids_list) < max_reasoning:
                reasoning_ids_list.append([0] * node_len)

            reasoning_ids_tensor = torch.tensor(
                [reasoning_ids_list], dtype=torch.long, device=self.device
            )
            reasoning_conf_tensor = torch.full(
                (1, max_reasoning),
                0.7, dtype=torch.float32, device=self.device,
            )

        return (
            evidence_ids_tensor,
            evidence_conf_tensor,
            anomaly_ids_tensor,
            anomaly_conf_tensor,
            reasoning_ids_tensor,
            reasoning_conf_tensor,
            composition_ids_tensor,
            composition_conf_tensor,
        )

    def _assess_complexity(
        self,
        graph_cond: dict[str, torch.Tensor],
        force_thinking_mode: Optional[str] = None,
    ) -> Optional[Any]:
        """Use ThinkingToggle to assess the complexity of the input.

        Args:
            graph_cond: Graph conditioning dict from encoder.
            force_thinking_mode: Force 'thinking' or 'non_thinking'.

        Returns:
            ThinkingAssessment or None if not available.
        """
        if not self._has_thinking_toggle:
            return None

        from diffusion_llm.model.thinking_toggle import ThinkingMode

        # Build a hidden-state-like tensor from graph conditioning
        # for the ThinkingToggle to assess
        graph_values = graph_cond.get("values")
        if graph_values is None:
            return None

        # Reshape to (batch, seq, d_model) if needed
        if graph_values.dim() == 2:
            graph_values = graph_values.unsqueeze(0)

        force_mode = None
        if force_thinking_mode == "thinking":
            force_mode = ThinkingMode.THINKING
        elif force_thinking_mode == "non_thinking":
            force_mode = ThinkingMode.NON_THINKING

        try:
            assessment = self.model.thinking_toggle(
                graph_values, force_mode=force_mode
            )
            return assessment
        except Exception as e:
            logger.warning("ThinkingToggle assessment failed: %s", e)
            return None

    def _should_use_mcts(
        self,
        use_mcts: Optional[bool],
        assessment: Optional[Any],
        method: str,
    ) -> bool:
        """Determine whether MCTS should be used.

        Logic:
        - If use_mcts is explicitly True/False, use that.
        - If use_mcts is None (auto), use MCTS when:
          - ThinkingToggle is in THINKING mode, AND
          - The task type is REASONING or ANOMALY_RESOLUTION, AND
          - MCTS module is available
        """
        if not self._has_mcts:
            return False

        if use_mcts is not None:
            return use_mcts

        # Auto-decide based on ThinkingToggle
        if assessment is None:
            return False

        from diffusion_llm.model.thinking_toggle import (
            ThinkingMode,
            TaskType,
        )

        if assessment.mode != ThinkingMode.THINKING:
            return False

        # Only use MCTS for reasoning-heavy task types
        if assessment.dominant_task in (
            TaskType.REASONING,
            TaskType.ANOMALY_RESOLUTION,
        ):
            return True

        return False

    def _run_mcts_reasoning(
        self,
        graph_cond: dict[str, torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        """Run MCTS reasoning on graph conditioning.

        Args:
            graph_cond: Graph conditioning dict from encoder.

        Returns:
            Dict with MCTS info, or None if MCTS failed.
        """
        graph_values = graph_cond.get("values")
        if graph_values is None:
            return None

        # Reshape for MCTS input
        if graph_values.dim() == 2:
            graph_values = graph_values.unsqueeze(0)

        try:
            action_probs, info = self.model.mcts_reasoner(graph_values)
            return {
                "action_probs_mean": action_probs.mean().item(),
                "total_simulations": info.get("total_simulations", 0),
                "root_value": info.get("root_value", 0.0),
                "entropy": info.get("entropy", 0.0),
            }
        except Exception as e:
            logger.warning("MCTS reasoning failed: %s", e)
            return None

    # ================================================================
    # Batch generation
    # ================================================================

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

    # ================================================================
    # Memory management
    # ================================================================

    def clear_memory(self) -> None:
        """Clear the model's dual memory system.

        Useful between independent generation sessions.
        """
        if self._has_dual_memory:
            self.model.memory_clear()
            logger.info("Dual memory cleared.")

    def get_memory_stats(self) -> Dict[str, object]:
        """Get current memory statistics.

        Returns:
            Dict with memory stats, or empty dict if memory disabled.
        """
        if self._has_dual_memory:
            return self.model.memory_stats()
        return {}

    # ================================================================
    # Convenience methods
    # ================================================================

    def generate_fast(
        self,
        trigger: str = "",
        **kwargs,
    ) -> GenerationResult:
        """Generate with fastest settings (non-thinking, anchored, minimal steps).

        Convenience wrapper for quick generation.

        Args:
            trigger: The trigger question or topic.
            **kwargs: Additional arguments passed to generate().

        Returns:
            GenerationResult with the narrative.
        """
        return self.generate(
            trigger=trigger,
            method="anchored",
            force_thinking_mode="non_thinking",
            use_mcts=False,
            n_steps=2,
            **kwargs,
        )

    def generate_deep(
        self,
        trigger: str = "",
        **kwargs,
    ) -> GenerationResult:
        """Generate with deepest reasoning (thinking, MCTS, more steps).

        Convenience wrapper for complex reasoning tasks.

        Args:
            trigger: The trigger question or topic.
            **kwargs: Additional arguments passed to generate().

        Returns:
            GenerationResult with the narrative.
        """
        method = "anchored" if self._has_anchored_decoder else "ddim"
        return self.generate(
            trigger=trigger,
            method=method,
            force_thinking_mode="thinking",
            use_mcts=True,
            n_steps=5 if method == "anchored" else 100,
            **kwargs,
        )
