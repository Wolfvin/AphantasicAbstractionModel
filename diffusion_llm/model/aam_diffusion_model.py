"""
AAM Diffusion LLM — Complete Model (v2.0)

Combines the Diffusion Transformer, Graph Encoder, and Noise Scheduler
into a single, unified model for training and inference.

v2.0 Upgrades:
    - ContinuousOutputHead (Anchored Decoder) replaces lm_head for
      2-3 step refinement instead of 50-step DDPM/DDIM
    - EvoformerManager for iterative bidirectional feedback
    - DualMemorySystem for long narrative generation
    - ThinkingToggle for adaptive compute (thinking vs non-thinking)
    - FlowMatchingDecoder as alternative sampling method
    - MCTSReasoner for complex reasoning tasks
    - Full backward compatibility (use_anchored_decoder=False)

Architecture:
    ┌──────────────────────────────────────────────────┐
    │  AAM Diffusion Model v2.0 (The Body)              │
    │                                                   │
    │  Input:                                           │
    │    - Token IDs (text)                             │
    │    - Graph conditioning (evidence, compositions,  │
    │      confidence, anomalies, reasoning chains)     │
    │                                                   │
    │  Training Process:                                │
    │    1. Tokenize text → embeddings                  │
    │    2. Sample random timestep t                    │
    │    3. Add noise: x_t = schedule.add_noise(x_0, t) │
    │    4. Encode graph conditioning                   │
    │    5. Predict noise: eps = transformer(x_t, t, c) │
    │    6. [Optional] Evoformer bidirectional feedback  │
    │    7. Compute loss: L = MSE(eps, eps_target)      │
    │                                                   │
    │  Inference Process (v2.0 Anchored):               │
    │    1. Encode graph conditioning                   │
    │    2. Transformer produces initial prediction     │
    │    3. Anchored Decoder refines in 2-3 steps       │
    │    4. Convert to tokens via ContinuousOutputHead   │
    │                                                   │
    │  Inference Process (Legacy DDPM/DDIM):            │
    │    1. Start from pure noise x_T                   │
    │    2. Encode graph conditioning                   │
    │    3. For t = T, T-1, ..., 1:                     │
    │       a. Predict noise: eps = transformer(x_t, t) │
    │       b. Denoise: x_{t-1} = schedule.step(eps)   │
    │    4. Decode final x_0 → text tokens              │
    │                                                   │
    │  Key Constraint:                                  │
    │    The model CANNOT generate information not       │
    │    present in the graph conditioning. It can only  │
    │    ARRANGE what the graph knows into sentences.    │
    │                                                   │
    │  Analogi: Jin Soun (mind/graph) + tubuhnya         │
    │  (this model). Tubuhnya hanya bisa mengucapkan    │
    │  apa yang dipikirkannya — tidak bisa mengarang.   │
    └──────────────────────────────────────────────────┘

Analogi: Ini adalah seluruh "tubuh" Jin Soun — bukan hanya
ototnya (transformer), tapi juga sistem saraf (graph encoder),
kemampuan untuk memperbaiki diri (diffusion denoising), dan
di v2.0: pikiran sadar (Evoformer), ingatan jangka panjang
(DualMemory), kemampuan berpikir adaptif (ThinkingToggle),
dan penalaran mendalam (MCTS).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from diffusion_llm.config.model_config import AamDiffusionConfig
from diffusion_llm.model.noise_scheduler import NoiseScheduler
from diffusion_llm.model.graph_encoder import GraphConditioningEncoder
from diffusion_llm.model.diffusion_transformer import DiffusionTransformer

logger = logging.getLogger(__name__)


class AamDiffusionModel(nn.Module):
    """Complete AAM Diffusion LLM model (v2.0).

    Combines:
    - DiffusionTransformer: Core denoising network
    - GraphConditioningEncoder: Encodes graph structure for conditioning
    - NoiseScheduler: Manages the diffusion process
    - [v2.0] ContinuousOutputHead: Anchored decoder for 2-3 step refinement
    - [v2.0] EvoformerManager: Iterative bidirectional feedback
    - [v2.0] DualMemorySystem: Working + long-term memory for narratives
    - [v2.0] ThinkingToggle: Adaptive compute based on input complexity
    - [v2.0] FlowMatchingDecoder: Alternative velocity-based sampling
    - [v2.0] MCTSReasoner: Tree search for complex reasoning

    This model is designed to be trained on Graph→Narrative pairs,
    where the graph data comes from the RSVS Knowledge Graph and
    the narrative is the target natural language output.

    Args:
        config: AamDiffusionConfig with all hyperparameters.
    """

    def __init__(self, config: AamDiffusionConfig):
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------
        # Feature flags — use getattr for backward compatibility so old
        # configs without the new fields still work.
        # ----------------------------------------------------------------
        self.use_anchored_decoder = getattr(config, "use_anchored_decoder", False)
        self.use_evoformer = getattr(config, "use_evoformer", False)
        self.use_dual_memory = getattr(config, "use_dual_memory", False)
        self.use_thinking_toggle = getattr(config, "use_thinking_toggle", False)
        self.use_flow_matching = getattr(config, "use_flow_matching", False)
        self.use_mcts = getattr(config, "use_mcts", False)

        # ----------------------------------------------------------------
        # Core components (always present)
        # ----------------------------------------------------------------
        self.noise_scheduler = NoiseScheduler(
            n_timesteps=config.diffusion.n_timesteps,
            schedule_type=config.diffusion.schedule_type,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            prediction_type=config.diffusion.prediction_type,
        )

        self.graph_encoder = GraphConditioningEncoder(
            config=config.graph_encoder,
            vocab_size=config.model.vocab_size,
        )
        # Align graph encoder output dim with transformer's d_model
        self.graph_encoder.set_output_dim(config.model.d_model)

        self.transformer = DiffusionTransformer(config.model)

        # ----------------------------------------------------------------
        # Output head — v2.0 ContinuousOutputHead or legacy lm_head
        # ----------------------------------------------------------------
        if self.use_anchored_decoder:
            from diffusion_llm.experimental.anchored_decoder import (
                ContinuousOutputHead,
                AnchoredDecoderConfig,
            )

            decoder_config = getattr(config, "anchored_decoder", None)
            if decoder_config is None:
                decoder_config = AnchoredDecoderConfig(
                    d_model=config.model.d_model,
                    d_vocab=config.model.vocab_size,
                )
            self.output_head = ContinuousOutputHead(
                d_model=config.model.d_model,
                d_vocab=config.model.vocab_size,
                decoder_config=decoder_config,
            )
        else:
            # Legacy: simple linear head with tied weights
            self.lm_head = nn.Linear(
                config.model.d_model, config.model.vocab_size, bias=False
            )
            self.lm_head.weight = self.transformer.token_embedding.weight

        # ----------------------------------------------------------------
        # Optional v2.0 modules — lazy imports
        # ----------------------------------------------------------------
        if self.use_evoformer:
            from diffusion_llm.experimental.evoformer import EvoformerManager, EvoformerConfig

            evoformer_config = getattr(config, "evoformer", None)
            if evoformer_config is None:
                evoformer_config = EvoformerConfig(d_model=config.model.d_model)
            else:
                # Sync d_model with the model's actual d_model
                evoformer_config.d_model = config.model.d_model
            self.evoformer = EvoformerManager(evoformer_config)

        if self.use_dual_memory:
            from diffusion_llm.experimental.dual_memory import (
                DualMemorySystem,
                DualMemoryConfig,
            )

            dual_memory_config = getattr(config, "dual_memory", None)
            if dual_memory_config is None:
                dual_memory_config = DualMemoryConfig(d_model=config.model.d_model)
            else:
                # Sync d_model with the model's actual d_model
                dual_memory_config.d_model = config.model.d_model
            self.dual_memory = DualMemorySystem(dual_memory_config)

        if self.use_thinking_toggle:
            from diffusion_llm.experimental.thinking_toggle import (
                ThinkingToggle,
                ThinkingMode,
            )

            thinking_config = getattr(config, "thinking_toggle", None)
            d_thinking = (
                thinking_config.d_model
                if thinking_config is not None
                else config.model.d_model
            )
            threshold = (
                thinking_config.threshold
                if thinking_config is not None
                else 0.5
            )
            self.thinking_toggle = ThinkingToggle(d_thinking, threshold)
            # Re-export for external use
            self.ThinkingMode = ThinkingMode

        if self.use_flow_matching:
            from diffusion_llm.experimental.flow_matching import FlowMatchingDecoder

            flow_config = getattr(config, "flow_matching", None)
            fm_d_model = (
                flow_config.d_model
                if flow_config is not None
                else config.model.d_model
            )
            fm_d_vocab = (
                flow_config.d_vocab
                if flow_config is not None
                else config.model.vocab_size
            )
            fm_num_steps = (
                flow_config.num_steps if flow_config is not None else 3
            )
            self.flow_matching_decoder = FlowMatchingDecoder(
                fm_d_model, fm_d_vocab, fm_num_steps
            )

        if self.use_mcts:
            from diffusion_llm.experimental.mcts import MCTSReasoner, MCTSConfig

            mcts_config = getattr(config, "mcts", None)
            if mcts_config is None:
                mcts_config = MCTSConfig()
            self.mcts_reasoner = MCTSReasoner(
                config.model.d_model, config=mcts_config
            )

        # ----------------------------------------------------------------
        # EMA model (for inference, updated during training)
        # ----------------------------------------------------------------
        self._ema_model: Optional[AamDiffusionModel] = None
        self._ema_decay = config.training.ema_decay

        # Build a summary of active modules
        active = []
        if self.use_anchored_decoder:
            active.append("AnchoredDecoder")
        if self.use_evoformer:
            active.append("Evoformer")
        if self.use_dual_memory:
            active.append("DualMemory")
        if self.use_thinking_toggle:
            active.append("ThinkingToggle")
        if self.use_flow_matching:
            active.append("FlowMatching")
        if self.use_mcts:
            active.append("MCTS")

        module_str = ", ".join(active) if active else "legacy"
        logger.info(
            "AamDiffusionModel v2.0 initialized: %s params, %s [modules: %s]",
            self._format_params(self.get_num_params()),
            config.model_name,
            module_str,
        )

    # ================================================================
    # Forward pass (training)
    # ================================================================

    def forward(
        self,
        token_ids: torch.Tensor,
        timestep: torch.Tensor,
        evidence_ids: Optional[torch.Tensor] = None,
        evidence_confidence: Optional[torch.Tensor] = None,
        evidence_timestamps: Optional[torch.Tensor] = None,
        composition_ids: Optional[torch.Tensor] = None,
        composition_confidence: Optional[torch.Tensor] = None,
        anomaly_ids: Optional[torch.Tensor] = None,
        anomaly_confidence: Optional[torch.Tensor] = None,
        anomaly_timestamps: Optional[torch.Tensor] = None,
        reasoning_ids: Optional[torch.Tensor] = None,
        reasoning_confidence: Optional[torch.Tensor] = None,
        source_trust: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        1. Get clean embeddings from token IDs
        2. Add noise at the given timestep
        3. Encode graph conditioning
        4. Predict noise via transformer
        5. [v2.0] Optionally apply Evoformer bidirectional feedback
        6. Return predicted noise (loss computed externally)

        Args:
            token_ids: Clean text token IDs, shape (batch, seq_len).
            timestep: Random timestep indices, shape (batch,).
            evidence_ids: Evidence node token IDs.
            evidence_confidence: Evidence confidence scores.
            evidence_timestamps: Evidence timestamps.
            composition_ids: Composition token IDs.
            composition_confidence: Composition confidence.
            anomaly_ids: Anomaly token IDs.
            anomaly_confidence: Anomaly confidence.
            anomaly_timestamps: Anomaly timestamps.
            reasoning_ids: Reasoning step token IDs.
            reasoning_confidence: Reasoning confidence.
            source_trust: Source trust score.

        Returns:
            Tuple of (predicted_noise, target_noise).
        """
        # Step 1: Get clean embeddings (x_0)
        x_0 = self.transformer.token_embedding(token_ids)

        # Step 2: Add noise
        noise = torch.randn_like(x_0)
        x_t = self.noise_scheduler.add_noise(x_0, noise, timestep)

        # Step 3: Encode graph conditioning
        batch_size = token_ids.shape[0]
        graph_cond = self.graph_encoder(
            evidence_ids=evidence_ids,
            evidence_confidence=evidence_confidence,
            evidence_timestamps=evidence_timestamps,
            composition_ids=composition_ids,
            composition_confidence=composition_confidence,
            anomaly_ids=anomaly_ids,
            anomaly_confidence=anomaly_confidence,
            anomaly_timestamps=anomaly_timestamps,
            reasoning_ids=reasoning_ids,
            reasoning_confidence=reasoning_confidence,
            source_trust=source_trust,
            batch_size=batch_size,
        )

        # Extract cross-attention keys/values from graph conditioning
        graph_keys = graph_cond.get("keys")
        graph_values = graph_cond.get("values")

        # [v2.0] Dual memory: enrich graph conditioning with memory
        if self.use_dual_memory:
            # Write current graph context to working memory
            if graph_values is not None:
                self.dual_memory.write(graph_values)
            # Read memory-augmented context
            if graph_keys is not None:
                graph_keys = self.dual_memory.read(graph_keys)
            if graph_values is not None:
                graph_values = self.dual_memory.read(graph_values)

        # Step 4: Predict noise via transformer
        predicted = self.transformer(
            x_t=x_t,
            t=timestep,
            graph_keys=graph_keys,
            graph_values=graph_values,
        )

        # [v2.0] Evoformer: bidirectional feedback between
        # transformer output and graph conditioning
        if self.use_evoformer:
            # Level 2: Bidirectional token update
            predicted = self.evoformer.bidirectional_token_update(predicted)

            # Level 3: Decoder-predict feedback — graph output refines prediction
            if graph_values is not None:
                # Use mean-pooled graph values as the "decoder output"
                graph_pooled = graph_values.mean(dim=1, keepdim=True).expand_as(
                    predicted
                )
                predicted = self.evoformer.apply_decoder_feedback(
                    predicted, graph_pooled
                )

            # Level 4: Prediction recycling — predicted output refines context
            if self.use_anchored_decoder and hasattr(self, "output_head"):
                # Get preliminary logits for prediction recycling
                with torch.no_grad():
                    prelim_vectors = self.output_head.get_continuous_vectors(predicted)
                predicted = self.evoformer.apply_prediction_recycling(
                    predicted, prelim_vectors
                )

        return predicted, noise

    # ================================================================
    # Loss computation
    # ================================================================

    def compute_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Supports different loss types and weighting strategies.

        Args:
            predicted: Model output (predicted noise/x0/v).
            target: Target (actual noise/x0/v).
            timestep: Timestep indices for loss weighting.

        Returns:
            Scalar loss value.
        """
        # Base loss
        if self.config.diffusion.loss_type == "mse":
            loss = nn.functional.mse_loss(predicted, target, reduction="none")
        elif self.config.diffusion.loss_type == "mae":
            loss = nn.functional.l1_loss(predicted, target, reduction="none")
        elif self.config.diffusion.loss_type == "huber":
            loss = nn.functional.smooth_l1_loss(predicted, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {self.config.diffusion.loss_type}")

        # Average over feature dimension
        loss = loss.mean(dim=-1)  # (batch, seq_len)

        # Apply loss weighting
        if self.config.diffusion.loss_weighting == "min_snr":
            loss = self._apply_min_snr_weighting(loss, timestep)
        elif self.config.diffusion.loss_weighting == "p2":
            loss = self._apply_p2_weighting(loss, timestep)

        # Average over sequence and batch
        return loss.mean()

    def _apply_min_snr_weighting(
        self,
        loss: torch.Tensor,
        timestep: torch.Tensor,
        gamma: float = 5.0,
    ) -> torch.Tensor:
        """Apply Min-SNR weighting strategy.

        Weights the loss by min(SNR, gamma) / SNR, where
        SNR = alpha_bar / (1 - alpha_bar).

        This helps balance the loss across timesteps, preventing
        high-noise steps from dominating.

        Args:
            loss: Unweighted loss.
            timestep: Timestep indices.
            gamma: SNR clipping value.

        Returns:
            Weighted loss.
        """
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(loss.device)
        snr = alpha_bar[timestep] / (1 - alpha_bar[timestep] + 1e-8)
        weight = torch.clamp(snr, max=gamma) / (snr + 1e-8)
        # Expand weight to match loss shape
        weight = weight.unsqueeze(-1).expand_as(loss)
        return loss * weight

    def _apply_p2_weighting(
        self,
        loss: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply P2 weighting strategy.

        weight = 1 / (SNR^gamma + k)

        Args:
            loss: Unweighted loss.
            timestep: Timestep indices.

        Returns:
            Weighted loss.
        """
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(loss.device)
        snr = alpha_bar[timestep] / (1 - alpha_bar[timestep] + 1e-8)
        gamma = self.config.diffusion.p2_gamma
        k = self.config.diffusion.p2_k
        weight = 1.0 / (snr ** gamma + k)
        weight = weight.unsqueeze(-1).expand_as(loss)
        return loss * weight

    # ================================================================
    # Sampling / Inference
    # ================================================================

    @torch.no_grad()
    def sample(
        self,
        graph_cond: dict[str, torch.Tensor],
        n_steps: Optional[int] = None,
        method: str = "ddim",
        shape: Optional[tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples via iterative denoising.

        This is the INFERENCE method. Supports multiple sampling
        strategies in v2.0:

        - "anchored": Uses ContinuousOutputHead for 2-3 step refinement
          (fastest, starts from graph-conditioned prediction)
        - "flow_matching": Uses FlowMatchingDecoder for velocity-based
          sampling (2-3 steps)
        - "ddpm": Legacy full DDPM sampling (many steps)
        - "ddim": Legacy DDIM sampling (fewer steps, deterministic)

        Args:
            graph_cond: Graph conditioning dict from GraphConditioningEncoder.
            n_steps: Number of denoising steps. Uses config if None.
            method: Sampling method — 'anchored', 'flow_matching',
                'ddpm', or 'ddim'.
            shape: Shape of the output (batch, seq_len, d_model).
            device: Device to generate on.
            temperature: Sampling temperature.

        Returns:
            Denoised embeddings of shape (batch, seq_len, d_model).
        """
        if n_steps is None:
            n_steps = self.config.diffusion.n_inference_steps
        if device is None:
            device = next(self.parameters()).device
        if shape is None:
            shape = (1, self.config.model.max_seq_len, self.config.model.d_model)

        # Get graph conditioning
        graph_keys = graph_cond.get("keys")
        graph_values = graph_cond.get("values")

        # [v2.0] Dual memory: augment graph conditioning with memory
        if self.use_dual_memory:
            if graph_values is not None:
                self.dual_memory.write(graph_values)
            if graph_keys is not None:
                graph_keys = self.dual_memory.read(graph_keys)
            if graph_values is not None:
                graph_values = self.dual_memory.read(graph_values)

        # ----------------------------------------------------------
        # METHOD: Anchored Decoder (2-3 step refinement)
        # ----------------------------------------------------------
        if method == "anchored" and hasattr(self, "output_head"):
            return self._sample_anchored(
                graph_keys, graph_values, shape, device, n_steps, temperature
            )

        # ----------------------------------------------------------
        # METHOD: Flow Matching Decoder
        # ----------------------------------------------------------
        if method == "flow_matching" and hasattr(self, "flow_matching_decoder"):
            return self._sample_flow_matching(
                graph_keys, graph_values, shape, device
            )

        # ----------------------------------------------------------
        # METHOD: Legacy DDPM / DDIM
        # ----------------------------------------------------------
        return self._sample_legacy(
            graph_keys, graph_values, shape, device, n_steps, method
        )

    def _sample_anchored(
        self,
        graph_keys: Optional[torch.Tensor],
        graph_values: Optional[torch.Tensor],
        shape: tuple[int, ...],
        device: torch.device,
        n_steps: int,
        temperature: float,
    ) -> torch.Tensor:
        """Anchored decoding: start from transformer prediction, refine 2-3 steps.

        Key insight: Instead of starting from noise and denoising for 50+
        steps, we use the transformer's graph-conditioned prediction as an
        anchor and refine it with the AnchoredDiffusionDecoder.
        """
        # Step 1: Get an initial prediction from the transformer
        # Use a low-noise timestep so the transformer gives a meaningful
        # starting point (t=0 would be ideal but we use a small t for
        # stability with the noise scheduler)
        batch_size = shape[0]
        t_init = torch.full(
            (batch_size,), 0, device=device, dtype=torch.long
        )

        # Start from a small amount of structured noise
        x = torch.randn(shape, device=device) * 0.1

        # Single transformer forward pass to get the initial anchor
        initial_pred = self.transformer(
            x_t=x, t=t_init,
            graph_keys=graph_keys,
            graph_values=graph_values,
        )

        # [v2.0] Evoformer feedback on initial prediction
        if self.use_evoformer:
            initial_pred = self.evoformer.bidirectional_token_update(initial_pred)
            if graph_values is not None:
                graph_pooled = graph_values.mean(dim=1, keepdim=True).expand_as(
                    initial_pred
                )
                initial_pred = self.evoformer.apply_decoder_feedback(
                    initial_pred, graph_pooled
                )

        # [v2.0] ThinkingToggle: determine refinement depth
        refine_steps = n_steps
        if self.use_thinking_toggle:
            assessment = self.thinking_toggle(initial_pred)
            # Scale refinement steps by depth multiplier
            depth_mult = assessment.depth_multiplier.mean().item()
            refine_steps = max(2, min(5, int(3 * depth_mult)))
            logger.debug(
                "ThinkingToggle: mode=%s, depth_mult=%.2f, refine_steps=%d",
                assessment.mode.value,
                depth_mult,
                refine_steps,
            )

        # Step 2: Refine with Anchored Decoder
        # The output_head internally does disambiguation + coherence
        # + optional evoformer feedback in 2-3 steps
        graph_context = graph_values.mean(dim=1) if graph_values is not None else None
        logits, info = self.output_head(
            initial_pred,
            use_diffusion=True,
            context=graph_context,
        )

        # The output_head gives us logits; we need to project back to
        # embedding space for the final embeddings_to_tokens step.
        # Use the token embedding matrix to convert logits → embeddings
        logits_scaled = logits / temperature
        probs = torch.softmax(logits_scaled, dim=-1)
        embeddings = torch.matmul(
            probs, self.transformer.token_embedding.weight
        )

        logger.debug(
            "Anchored sampling: %d refine steps, delta=%.4f",
            info.get("n_refine_steps", refine_steps),
            info.get("refinement_delta", 0.0),
        )

        return embeddings

    def _sample_flow_matching(
        self,
        graph_keys: Optional[torch.Tensor],
        graph_values: Optional[torch.Tensor],
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Flow matching sampling: velocity-based 2-3 step refinement."""
        batch_size = shape[0]

        # Step 1: Get initial hidden state from transformer
        t_init = torch.full(
            (batch_size,), 0, device=device, dtype=torch.long
        )
        x = torch.randn(shape, device=device) * 0.1

        initial_pred = self.transformer(
            x_t=x, t=t_init,
            graph_keys=graph_keys,
            graph_values=graph_values,
        )

        # [v2.0] Evoformer feedback on initial prediction
        if self.use_evoformer:
            initial_pred = self.evoformer.bidirectional_token_update(initial_pred)
            if graph_values is not None:
                graph_pooled = graph_values.mean(dim=1, keepdim=True).expand_as(
                    initial_pred
                )
                initial_pred = self.evoformer.apply_decoder_feedback(
                    initial_pred, graph_pooled
                )

        # Step 2: Flow matching refinement
        flow_output = self.flow_matching_decoder(initial_pred)

        # Convert flow-matched logits back to embedding space
        probs = torch.softmax(flow_output.refined_logits, dim=-1)
        embeddings = torch.matmul(
            probs, self.transformer.token_embedding.weight
        )

        logger.debug(
            "Flow matching sampling: %d steps",
            flow_output.num_steps,
        )

        return embeddings

    def _sample_legacy(
        self,
        graph_keys: Optional[torch.Tensor],
        graph_values: Optional[torch.Tensor],
        shape: tuple[int, ...],
        device: torch.device,
        n_steps: int,
        method: str,
    ) -> torch.Tensor:
        """Legacy DDPM/DDIM sampling (v1.0 compatible)."""
        # Start from pure noise
        x = torch.randn(shape, device=device)

        if method == "ddpm":
            # Full DDPM sampling
            for t in reversed(range(self.config.diffusion.n_timesteps)):
                t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
                predicted = self.transformer(
                    x_t=x, t=t_tensor,
                    graph_keys=graph_keys,
                    graph_values=graph_values,
                )

                # [v2.0] Evoformer feedback per step (expensive, only if enabled)
                if self.use_evoformer:
                    predicted = self.evoformer.bidirectional_token_update(predicted)

                x = self.noise_scheduler.step_ddpm(predicted, x, t_tensor)

        elif method == "ddim":
            # Fast DDIM sampling
            timesteps = self.noise_scheduler.get_timestep_schedule(n_steps)
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0

                t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
                predicted = self.transformer(
                    x_t=x, t=t_tensor,
                    graph_keys=graph_keys,
                    graph_values=graph_values,
                )

                # [v2.0] Evoformer feedback per step
                if self.use_evoformer:
                    predicted = self.evoformer.bidirectional_token_update(predicted)

                x = self.noise_scheduler.step_ddim(
                    predicted, x, t, t_prev,
                    eta=self.config.diffusion.eta_ddim,
                )
        else:
            raise ValueError(
                f"Unknown sampling method: {method}. "
                f"Use 'anchored', 'flow_matching', 'ddpm', or 'ddim'."
            )

        return x

    # ================================================================
    # Embedding → Token conversion
    # ================================================================

    def embeddings_to_tokens(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        graph_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert continuous embeddings to discrete token IDs.

        This is the final step of generation — project embeddings
        to vocabulary logits and sample tokens.

        v2.0: When ContinuousOutputHead is available, it uses the
        anchored decoder for refined logits. Otherwise falls back
        to the standard lm_head.

        Args:
            embeddings: Denoised embeddings of shape (batch, seq_len, d_model).
            temperature: Sampling temperature.
            top_k: Top-k sampling cutoff.
            graph_context: Optional graph conditioning for anchored decoder.

        Returns:
            Token IDs of shape (batch, seq_len).
        """
        if hasattr(self, "output_head"):
            # v2.0: Use anchored decoder for refined logit prediction
            logits, info = self.output_head(
                embeddings, use_diffusion=True, context=graph_context
            )
            logits = logits / temperature
        else:
            # Legacy: simple linear projection
            logits = self.lm_head(embeddings) / temperature

        # Top-k sampling
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_k_values, dim=-1)
            sampled_indices = torch.multinomial(
                probs.view(-1, top_k), 1
            ).view(logits.shape[0], logits.shape[1])
            token_ids = top_k_indices.gather(
                -1, sampled_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            token_ids = torch.argmax(logits, dim=-1)

        return token_ids

    # ================================================================
    # ThinkingToggle integration
    # ================================================================

    def assess_thinking(
        self, hidden_states: torch.Tensor, force_mode=None
    ) -> Optional[Any]:
        """Assess whether the input needs deep thinking or quick response.

        Only available when use_thinking_toggle=True.

        Args:
            hidden_states: Hidden states to assess, shape (batch, seq_len, d_model).
            force_mode: Optional ThinkingMode to override the assessment.

        Returns:
            ThinkingAssessment if ThinkingToggle is enabled, else None.
        """
        if not self.use_thinking_toggle:
            return None
        return self.thinking_toggle(hidden_states, force_mode=force_mode)

    # ================================================================
    # MCTS integration
    # ================================================================

    def reason_with_mcts(
        self,
        hidden_states: torch.Tensor,
        num_simulations: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, Dict[str, Any]]]:
        """Run MCTS reasoning on hidden states.

        Only available when use_mcts=True.

        Args:
            hidden_states: Hidden states to reason about.
            num_simulations: Override number of MCTS simulations.

        Returns:
            Tuple of (action_probs, info_dict) if MCTS enabled, else None.
        """
        if not self.use_mcts:
            return None
        return self.mcts_reasoner(hidden_states, num_simulations=num_simulations)

    # ================================================================
    # Dual Memory management
    # ================================================================

    def memory_consolidate(self) -> None:
        """Consolidate working memory into long-term memory.

        Only available when use_dual_memory=True.
        """
        if self.use_dual_memory:
            self.dual_memory.consolidate()

    def memory_clear(self) -> None:
        """Clear working memory.

        Only available when use_dual_memory=True.
        """
        if self.use_dual_memory:
            self.dual_memory.clear()

    def memory_stats(self) -> Dict[str, object]:
        """Get memory system statistics.

        Returns:
            Dict with memory stats, or empty dict if DualMemory disabled.
        """
        if self.use_dual_memory:
            return self.dual_memory.get_stats()
        return {}

    # ================================================================
    # Evoformer statistics
    # ================================================================

    def evoformer_stats(self) -> Dict[str, object]:
        """Get Evoformer feedback statistics.

        Returns:
            Dict with evoformer stats, or empty dict if Evoformer disabled.
        """
        if self.use_evoformer:
            return self.evoformer.get_stats()
        return {}

    # ================================================================
    # Utility methods
    # ================================================================

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _format_params(n: int) -> str:
        """Format parameter count for display."""
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.1f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Output file path.
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> AamDiffusionModel:
        """Load model from checkpoint.

        Supports both v2.0 and v1.0 checkpoints. Missing v2.0 config
        fields are filled with defaults (disabled), ensuring backward
        compatibility.

        Args:
            path: Checkpoint file path.
            device: Device to load to.

        Returns:
            Loaded AamDiffusionModel.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config_dict = checkpoint.get("config", {})
        if isinstance(config_dict, dict):
            config = AamDiffusionConfig()
            # Try to reconstruct config from dict
            try:
                from diffusion_llm.config.model_config import (
                    ModelConfig, DiffusionConfig, GraphEncoderConfig,
                    TokenizerConfig, TrainingConfig, InferenceConfig,
                )
                config = AamDiffusionConfig(
                    model=ModelConfig(**config_dict.get("model", {})),
                    diffusion=DiffusionConfig(**config_dict.get("diffusion", {})),
                    graph_encoder=GraphEncoderConfig(**config_dict.get("graph_encoder", {})),
                    tokenizer=TokenizerConfig(**config_dict.get("tokenizer", {})),
                    training=TrainingConfig(**config_dict.get("training", {})),
                    inference=InferenceConfig(**config_dict.get("inference", {})),
                    model_name=config_dict.get("model_name", "aam-diffusion-v0.1"),
                    output_dir=config_dict.get("output_dir", "./output"),
                    seed=config_dict.get("seed", 42),
                )
            except Exception:
                logger.warning("Could not reconstruct config from checkpoint, using defaults")
        else:
            config = config_dict

        # v2.0 config fields — attach from checkpoint dict if present
        # so the model initializes optional modules correctly
        for flag in [
            "use_anchored_decoder", "use_evoformer", "use_dual_memory",
            "use_thinking_toggle", "use_flow_matching", "use_mcts",
        ]:
            if flag not in config_dict:
                # Old checkpoint — ensure the flag is False
                if not hasattr(config, flag):
                    setattr(config, flag, False)

        # Attach sub-configs if present in checkpoint
        for sub_key in [
            "anchored_decoder", "evoformer", "dual_memory",
            "thinking_toggle", "flow_matching", "mcts",
        ]:
            if sub_key in config_dict and not hasattr(config, sub_key):
                setattr(config, sub_key, config_dict[sub_key])

        model = cls(config)

        # Load state dict with partial matching for backward compatibility
        state_dict = checkpoint["model_state_dict"]
        model_state = model.state_dict()

        # Separate keys that match vs. don't match
        matched = {k: v for k, v in state_dict.items() if k in model_state}
        missing = [k for k in model_state if k not in state_dict]
        unexpected = [k for k in state_dict if k not in model_state]

        if missing:
            logger.info(
                "Loading checkpoint: %d keys missing (new v2.0 modules), "
                "will use random init for those.",
                len(missing),
            )
        if unexpected:
            logger.info(
                "Loading checkpoint: %d unexpected keys (legacy modules).",
                len(unexpected),
            )

        model.load_state_dict(matched, strict=False)
        model.to(device)
        logger.info("Model loaded from %s", path)
        return model
