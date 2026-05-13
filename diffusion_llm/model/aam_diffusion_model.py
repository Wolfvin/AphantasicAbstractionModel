"""
AAM Diffusion LLM — Complete Model

Combines the Diffusion Transformer, Graph Encoder, and Noise Scheduler
into a single, unified model for training and inference.

This is the "body" of AAM — the specialized sentence composer that
takes graph conditioning as input and produces coherent narratives
through iterative denoising.

Architecture:
    ┌──────────────────────────────────────────────────┐
    │  AAM Diffusion Model (The Body)                   │
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
    │    6. Compute loss: L = MSE(eps, eps_target)      │
    │                                                   │
    │  Inference Process:                               │
    │    1. Start from pure noise x_T                   │
    │    2. Encode graph conditioning                   │
    │    3. For t = T, T-1, ..., 1:                     │
    │       a. Predict noise: eps = transformer(x_t, t) │
    │       b. Denoise: x_{t-1} = schedule.step(eps)   │
    │    4. Decode final x_0 → text tokens              │
    │    5. Detokenize → natural language narrative     │
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
ototnya (transformer), tapi juga sistem saraf (graph encoder)
dan kemampuan untuk memperbaiki diri (diffusion denoising).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from diffusion_llm.config.model_config import AamDiffusionConfig
from diffusion_llm.model.noise_scheduler import NoiseScheduler
from diffusion_llm.model.graph_encoder import GraphConditioningEncoder
from diffusion_llm.model.diffusion_transformer import DiffusionTransformer

logger = logging.getLogger(__name__)


class AamDiffusionModel(nn.Module):
    """Complete AAM Diffusion LLM model.

    Combines:
    - DiffusionTransformer: Core denoising network
    - GraphConditioningEncoder: Encodes graph structure for conditioning
    - NoiseScheduler: Manages the diffusion process

    This model is designed to be trained on Graph→Narrative pairs,
    where the graph data comes from the RSVS Knowledge Graph and
    the narrative is the target natural language output.

    Args:
        config: AamDiffusionConfig with all hyperparameters.
    """

    def __init__(self, config: AamDiffusionConfig):
        super().__init__()
        self.config = config

        # Core components
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

        # Token-to-embedding projection (shared with transformer)
        # The transformer's token_embedding is used for both
        # encoding input text and decoding output text

        # Output head: project from d_model to vocab_size
        self.lm_head = nn.Linear(
            config.model.d_model, config.model.vocab_size, bias=False
        )

        # Tie weights between token embedding and LM head
        # This is standard practice and reduces parameter count
        self.lm_head.weight = self.transformer.token_embedding.weight

        # EMA model (for inference, updated during training)
        self._ema_model: Optional[AamDiffusionModel] = None
        self._ema_decay = config.training.ema_decay

        logger.info(
            "AamDiffusionModel initialized: %s params, %s",
            self._format_params(self.get_num_params()),
            config.model_name,
        )

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
    ) -> torch.Tensor:
        """Forward pass for training.

        1. Get clean embeddings from token IDs
        2. Add noise at the given timestep
        3. Encode graph conditioning
        4. Predict noise via transformer
        5. Return predicted noise (loss computed externally)

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
            Predicted noise tensor of shape (batch, seq_len, d_model).
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

        # Step 4: Predict noise via transformer
        predicted = self.transformer(
            x_t=x_t,
            t=timestep,
            graph_keys=graph_keys,
            graph_values=graph_values,
        )

        return predicted, noise

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

    @torch.no_grad()
    def sample(
        self,
        graph_cond: dict[str, torch.Tensor],
        n_steps: Optional[int] = None,
        method: str = "ddim",
        shape: Optional[tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate samples via iterative denoising.

        This is the INFERENCE method — start from pure noise and
        iteratively denoise to produce coherent text embeddings.

        Args:
            graph_cond: Graph conditioning dict from GraphConditioningEncoder.
            n_steps: Number of denoising steps. Uses config if None.
            method: Sampling method ('ddpm' or 'ddim').
            shape: Shape of the output (batch, seq_len, d_model).
            device: Device to generate on.

        Returns:
            Denoised embeddings of shape (batch, seq_len, d_model).
        """
        if n_steps is None:
            n_steps = self.config.diffusion.n_inference_steps
        if device is None:
            device = next(self.parameters()).device
        if shape is None:
            shape = (1, self.config.model.max_seq_len, self.config.model.d_model)

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Get graph conditioning
        graph_keys = graph_cond.get("keys")
        graph_values = graph_cond.get("values")

        if method == "ddpm":
            # Full DDPM sampling
            for t in reversed(range(self.config.diffusion.n_timesteps)):
                t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
                predicted = self.transformer(
                    x_t=x, t=t_tensor,
                    graph_keys=graph_keys,
                    graph_values=graph_values,
                )
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
                x = self.noise_scheduler.step_ddim(
                    predicted, x, t, t_prev,
                    eta=self.config.diffusion.eta_ddim,
                )

        return x

    def embeddings_to_tokens(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Convert continuous embeddings to discrete token IDs.

        This is the final step of generation — project embeddings
        to vocabulary logits and sample tokens.

        Args:
            embeddings: Denoised embeddings of shape (batch, seq_len, d_model).
            temperature: Sampling temperature.
            top_k: Top-k sampling cutoff.

        Returns:
            Token IDs of shape (batch, seq_len).
        """
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
            probs = torch.softmax(logits, dim=-1)
            token_ids = torch.argmax(logits, dim=-1)

        return token_ids

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
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        logger.info("Model loaded from %s", path)
        return model
