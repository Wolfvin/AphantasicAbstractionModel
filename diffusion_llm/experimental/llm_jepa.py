"""AAM Diffusion LLM — LLM-JEPA Training

Joint-Embedding Predictive Architecture (JEPA) adapted for the AAM
Diffusion LLM. Instead of predicting exact tokens, we predict the
REPRESENTATION of future tokens in continuous space.

Why JEPA for AAM?
    - Predictions live in continuous space (natural fit for diffusion)
    - No need for exact token matching (which is brittle for text)
    - The predictor learns "what comes next" in representation space
    - Perfect for AAM: predict graph-conditioned representations, not raw text

Architecture:
    ┌──────────────┐     ┌──────────────┐
    │  Online       │     │  Target       │
    │  Encoder      │     │  Encoder      │  ← EMA of online encoder
    │  (gradient)   │     │  (no grad)    │
    └──────┬───────┘     └──────┬───────┘
           │                     │
    current_repr            target_repr
    (masked positions       (at masked positions,
     replaced with          clean representation)
     [MASK] token)          │
           │                 │
           ▼                 │
    ┌──────────────┐         │
    │  JEPA        │         │
    │  Predictor   │         │
    └──────┬───────┘         │
           │                 │
    predicted_repr           │
           │                 │
           ▼                 ▼
       Loss = cosine_sim(predicted_repr, target_repr)

AAM-specific details:
    - The graph encoder provides conditioning that anchors representations
    - Evidence/anomaly/reasoning nodes shape the representation space
    - Masked positions correspond to narrative tokens conditioned on the graph
    - This makes the predictor learn graph-grounded continuations
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class JEPAConfig:
    """Configuration for LLM-JEPA training.

    Attributes:
        d_model: Model hidden dimension. Must match the encoder's d_model.
        prediction_horizon: How many steps ahead to predict. At horizon=1,
            the predictor learns the very next representation. At higher
            horizons, it learns to predict further into the future, which
            is useful for narrative planning in AAM.
        mask_ratio: Fraction of positions to mask in the input. Following
            I-JEPA, we use a moderate mask ratio (0.25-0.5) rather than
            the aggressive masking (0.75) used in MAE, because our
            predictor operates in representation space (not pixel space).
        ema_decay: Exponential moving average decay for the target encoder.
            Higher values = slower target update = more stable training.
            Typical range: 0.996-0.9996 (following I-JEPA / DINO v2).
        loss_type: Loss function for comparing predicted and target
            representations.
            - "mse": L2 distance (sensitive to scale)
            - "cosine": Cosine similarity (scale-invariant, recommended)
            - "smooth_l1": Huber loss (robust to outliers)
        n_predictor_layers: Number of transformer layers in the predictor.
        n_predictor_heads: Number of attention heads in the predictor.
        predictor_ff_dim: Feed-forward dimension in the predictor.
        warmup_steps: Number of EMA warmup steps (linear ramp from 0 to ema_decay).
        graph_conditioning: Whether to inject graph encoder output into
            the predictor as cross-attention context. This is AAM-specific:
            the graph provides evidence/anomaly/reasoning structure that
            strongly informs what representation should come next.
    """

    d_model: int = 768
    prediction_horizon: int = 1
    mask_ratio: float = 0.25
    ema_decay: float = 0.996
    loss_type: str = "cosine"
    n_predictor_layers: int = 6
    n_predictor_heads: int = 12
    predictor_ff_dim: int = 3072
    warmup_steps: int = 500
    graph_conditioning: bool = True


class FuturePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for future positions.

    When predicting `prediction_horizon` steps ahead, we need positional
    encodings for positions offset by the horizon. This module generates
    encodings at positions [seq_len, seq_len + prediction_horizon) so
    the predictor knows WHERE in the sequence it is predicting.
    """

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, seq_len: int, horizon: int, device: torch.device
    ) -> torch.Tensor:
        """Get positional encodings for future positions.

        Args:
            seq_len: Current sequence length.
            horizon: Prediction horizon offset.
            device: Target device.

        Returns:
            Positional encodings of shape (1, seq_len, d_model) for
            positions offset by `horizon`.
        """
        start = horizon
        end = start + seq_len
        end = min(end, self.max_len)
        return self.pe[start:end].unsqueeze(0).to(device)


class JEPAPredictor(nn.Module):
    """Predicts future representations from current ones using JEPA.

    This is the core of the JEPA framework: given the current context
    representations (with some positions masked), predict what the target
    encoder's representation would be at those masked positions, shifted
    forward by `prediction_horizon` steps.

    Architecture:
        1. Mask tokens are inserted at masked positions
        2. Future positional encodings signal "where" to predict
        3. Cross-attention between context and future position queries
           produces the predicted representation
        4. Optional graph conditioning via cross-attention

    AAM integration:
        - Graph encoder output serves as cross-attention context
        - Evidence/anomaly/reasoning nodes guide the prediction
        - The predictor learns graph-grounded representation dynamics
        - This is especially powerful for sentence arrangement, where
          the graph defines which sentences should follow which

    Args:
        config: JEPA configuration.
    """

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.prediction_horizon = config.prediction_horizon

        # Mask token embedding (learnable placeholder for masked positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Future positional encoding
        self.future_pe = FuturePositionalEncoding(config.d_model)

        # Input projection (for combining repr + positional info)
        self.input_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Predictor transformer layers
        # Ensure n_heads is a valid divisor of d_model
        n_heads = self._resolve_n_heads(config.d_model, config.n_predictor_heads)

        predictor_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=n_heads,
            dim_feedforward=config.predictor_ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor_layers = nn.TransformerDecoder(
            decoder_layer=predictor_layer,
            num_layers=config.n_predictor_layers,
        )

        # Graph conditioning cross-attention (AAM-specific)
        if config.graph_conditioning:
            self.graph_cross_attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=n_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.graph_gate = nn.Sequential(
                nn.Linear(config.d_model, 1, bias=False),
                nn.Sigmoid(),
            )
            self.graph_norm = nn.RMSNorm(config.d_model)

        # Output projection (predicted representation)
        self.output_proj = nn.Sequential(
            nn.RMSNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model, bias=False),
        )

    @staticmethod
    def _resolve_n_heads(d_model: int, requested_heads: int) -> int:
        """Resolve the number of attention heads to be a valid divisor of d_model.

        If requested_heads doesn't divide d_model evenly, find the closest
        valid divisor. This prevents assertion errors when d_model is not
        a multiple of the requested head count (e.g., d_model=256, heads=12).

        Args:
            d_model: Model hidden dimension.
            requested_heads: Desired number of heads.

        Returns:
            Valid number of heads that divides d_model evenly.
        """
        if d_model % requested_heads == 0:
            return requested_heads

        # Find all valid divisors
        valid_heads = [h for h in range(1, d_model + 1) if d_model % h == 0]

        # Find closest to requested
        best = min(valid_heads, key=lambda h: abs(h - requested_heads))
        return best

    def forward(
        self,
        current_repr: torch.Tensor,
        mask: torch.Tensor,
        graph_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future representations at masked positions.

        Args:
            current_repr: Current encoder representations of shape
                (batch, seq_len, d_model). Some positions correspond to
                visible tokens, others will be replaced by mask tokens.
            mask: Boolean mask of shape (batch, seq_len). True = this
                position is masked and should be predicted.
            graph_context: Optional graph encoder output of shape
                (batch, n_nodes, d_model). Used for AAM graph conditioning.

        Returns:
            Predicted representations at masked positions, of shape
            (batch, seq_len, d_model). Only the values at masked positions
            are meaningful; unmasked positions contain zeroed predictions.
        """
        batch_size, seq_len, d_model = current_repr.shape
        device = current_repr.device

        # Step 1: Replace masked positions with learnable mask tokens
        mask_expanded = mask.unsqueeze(-1).expand_as(current_repr)
        mask_tokens = self.mask_token.expand(batch_size, seq_len, d_model)
        masked_repr = torch.where(mask_expanded, mask_tokens, current_repr)

        # Step 2: Add future positional encoding
        future_pos = self.future_pe(seq_len, self.prediction_horizon, device)
        masked_repr = masked_repr + future_pos

        # Step 3: Project input
        x = self.input_proj(masked_repr)

        # Step 4: Apply predictor transformer (self-attention over sequence)
        # Use causal mask to prevent looking ahead
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )
        # TransformerDecoder needs a memory input; use the input itself
        x = self.predictor_layers(x, x, tgt_mask=causal_mask)

        # Step 5: Graph conditioning (AAM-specific)
        if graph_context is not None and self.config.graph_conditioning:
            graph_out, _ = self.graph_cross_attn(
                query=x, key=graph_context, value=graph_context
            )
            gate = self.graph_gate(x)
            x = self.graph_norm(x + gate * graph_out)

        # Step 6: Output projection
        predicted = self.output_proj(x)

        # Zero out predictions at non-masked positions
        predicted = predicted * mask_expanded.float()

        return predicted


class JEPATrainer:
    """Training loop for LLM-JEPA with EMA target encoder.

    The JEPA training paradigm:
        1. Online encoder processes the input (with some positions masked)
        2. Target encoder (EMA of online) processes the FULL input
        3. Predictor predicts target representations at masked positions
        4. Only predictor + online encoder get gradients
        5. Target encoder is updated via EMA

    Key insight for AAM: The graph-conditioned representations encode
    evidence/anomaly/reasoning structure. By predicting these in
    representation space (rather than token space), the model learns
    the DYNAMICS of narrative continuation grounded in the knowledge
    graph. This is more robust than token-level prediction because:
        - Similar narratives can have different surface tokens but
          similar representations
        - The graph provides strong priors for what should come next
        - Representation-level prediction naturally handles the
          one-to-many mapping from evidence to narrative

    Example usage:
        >>> config = JEPAConfig(d_model=768, prediction_horizon=1)
        >>> encoder = AamDiffusionModel(...)  # any encoder
        >>> trainer = JEPATrainer(config, encoder)
        >>> loss = trainer.train_step(token_ids, graph_context=graph_out)

    Args:
        config: JEPA training configuration.
        encoder: The main encoder model (e.g., AamDiffusionModel).
            Must produce representations of shape (batch, seq, d_model).
        optimizer: Optional pre-configured optimizer. If None, AdamW
            is created with default LR.
    """

    def __init__(
        self,
        config: JEPAConfig,
        encoder: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.device = next(encoder.parameters()).device

        # Create predictor
        self.predictor = JEPAPredictor(config).to(self.device)

        # Create EMA target encoder (no gradients)
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

        # Optimizer for online encoder + predictor (NOT target encoder)
        trainable_params = (
            list(encoder.parameters())
            + list(self.predictor.parameters())
        )
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=1e-4,
                betas=(0.9, 0.95),
                weight_decay=0.04,
            )

        # EMA schedule
        self.ema_decay = config.ema_decay
        self.warmup_steps = config.warmup_steps
        self.global_step = 0

        # Loss function selector
        self._loss_fn = self._get_loss_fn(config.loss_type)

        logger.info(
            "JEPATrainer initialized: d_model=%d, horizon=%d, mask_ratio=%.2f, "
            "ema_decay=%.4f, loss_type=%s, graph_conditioning=%s",
            config.d_model,
            config.prediction_horizon,
            config.mask_ratio,
            config.ema_decay,
            config.loss_type,
            config.graph_conditioning,
        )

    def _get_loss_fn(self, loss_type: str):
        """Return the appropriate loss function."""
        if loss_type == "mse":
            return self._mse_loss
        elif loss_type == "cosine":
            return self._cosine_loss
        elif loss_type == "smooth_l1":
            return self._smooth_l1_loss
        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type}. "
                f"Choose from: 'mse', 'cosine', 'smooth_l1'"
            )

    @staticmethod
    def _mse_loss(
        predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss averaged over masked positions."""
        diff = (predicted - target) ** 2
        diff = diff.sum(dim=-1)  # sum over d_model
        mask_float = mask.float()
        loss = (diff * mask_float).sum() / mask_float.sum().clamp(min=1)
        return loss

    @staticmethod
    def _cosine_loss(
        predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity loss (1 - cos_sim) averaged over masked positions.

        This is the recommended loss for JEPA because:
        - Scale-invariant: focuses on direction, not magnitude
        - More stable training than MSE for high-dim representations
        - Naturally encourages representations to be on the unit sphere
        """
        # Normalize predictions and targets
        pred_norm = F.normalize(predicted, dim=-1)
        tgt_norm = F.normalize(target, dim=-1)

        # Cosine similarity per position
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (batch, seq)

        # Loss = 1 - cosine_similarity (minimize to maximize similarity)
        loss_per_pos = 1.0 - cos_sim

        # Average only over masked positions
        mask_float = mask.float()
        loss = (loss_per_pos * mask_float).sum() / mask_float.sum().clamp(min=1)
        return loss

    @staticmethod
    def _smooth_l1_loss(
        predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Smooth L1 (Huber) loss averaged over masked positions."""
        diff = F.smooth_l1_loss(predicted, target, reduction="none")
        diff = diff.sum(dim=-1)  # sum over d_model
        mask_float = mask.float()
        loss = (diff * mask_float).sum() / mask_float.sum().clamp(min=1)
        return loss

    @torch.no_grad()
    def _get_ema_decay(self) -> float:
        """Get current EMA decay with warmup schedule.

        During warmup, the EMA decay ramps up linearly from 0 to the
        target ema_decay. This prevents the target encoder from being
        too far behind the online encoder early in training.
        """
        if self.global_step < self.warmup_steps:
            # Linear warmup from 0 to ema_decay
            return self.ema_decay * (self.global_step / max(self.warmup_steps, 1))
        return self.ema_decay

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """Update target encoder via EMA of online encoder.

        target_param = decay * target_param + (1 - decay) * online_param

        Only the online encoder (self.encoder) provides new weights;
        the target encoder never receives gradients directly.
        """
        decay = self._get_ema_decay()
        online_params = list(self.encoder.parameters())
        target_params = list(self.target_encoder.parameters())

        if len(online_params) != len(target_params):
            logger.warning(
                "Online encoder (%d params) and target encoder (%d params) "
                "parameter count mismatch. Skipping EMA update.",
                len(online_params),
                len(target_params),
            )
            return

        for t_param, o_param in zip(target_params, online_params):
            t_param.data.mul_(decay).add_(o_param.data, alpha=1.0 - decay)

    def _generate_mask(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Generate random mask for input positions.

        Following I-JEPA, we use random block masking rather than
        random individual token masking. This encourages the predictor
        to learn coherent multi-step predictions rather than
        memorizing individual token representations.

        For AAM, the blocks correspond to sentence-level spans,
        which aligns well with the sentence arrangement task.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Boolean mask of shape (batch, seq_len). True = masked.
        """
        n_masked = int(seq_len * self.config.mask_ratio)
        n_masked = max(1, n_masked)

        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # Random block masking: choose a random start and mask a contiguous block
        for b in range(batch_size):
            # Block length = sqrt(mask_ratio) * seq_len (following I-JEPA)
            block_len = max(1, int(math.sqrt(self.config.mask_ratio) * seq_len))
            n_blocks = max(1, n_masked // block_len)

            for _ in range(n_blocks):
                start = torch.randint(0, max(seq_len - block_len, 1), (1,)).item()
                end = min(start + block_len, seq_len)
                mask[b, start:end] = True

        return mask

    def train_step(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        graph_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Execute a single JEPA training step.

        Steps:
            1. Generate random mask over input positions
            2. Online encoder processes the full input
            3. Target encoder (EMA) processes the full input (no grad)
            4. Predictor predicts target representations at masked positions
            5. Compute loss between predicted and target representations
            6. Backprop through predictor + online encoder only
            7. EMA-update the target encoder

        Args:
            token_ids: Input token IDs of shape (batch, seq_len).
            attention_mask: Optional attention mask of shape (batch, seq_len).
            graph_context: Optional graph encoder output of shape
                (batch, n_nodes, d_model) for AAM graph conditioning.

        Returns:
            Dictionary of training metrics including:
            - "loss": Total JEPA loss
            - "cosine_sim": Mean cosine similarity at masked positions
            - "mask_ratio_actual": Actual fraction of positions masked
            - "ema_decay": Current EMA decay value
        """
        self.encoder.train()
        self.predictor.train()
        self.target_encoder.eval()

        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Step 1: Generate mask
        mask = self._generate_mask(batch_size, seq_len, device)
        if attention_mask is not None:
            # Don't mask padding positions
            mask = mask & attention_mask.bool()

        # Step 2: Online encoder forward pass (with gradient)
        online_repr = self._encode_with_encoder(
            self.encoder, token_ids, attention_mask
        )  # (batch, seq, d_model)

        # Step 3: Target encoder forward pass (no gradient, full input)
        with torch.no_grad():
            target_repr = self._encode_with_encoder(
                self.target_encoder, token_ids, attention_mask
            )  # (batch, seq, d_model)
            # Detach to be extra safe
            target_repr = target_repr.detach()

        # Step 4: Predictor predicts target repr at masked positions
        predicted_repr = self.predictor(
            current_repr=online_repr,
            mask=mask,
            graph_context=graph_context,
        )  # (batch, seq, d_model)

        # Step 5: Compute loss at masked positions only
        loss = self._loss_fn(predicted_repr, target_repr, mask)

        # Step 6: Backprop through predictor + online encoder
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.predictor.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        # Step 7: EMA update target encoder
        self._update_target_encoder()

        # Compute additional metrics
        with torch.no_grad():
            mask_float = mask.float()
            n_masked = mask_float.sum().item()
            total_positions = mask_float.numel()
            mask_ratio_actual = n_masked / max(total_positions, 1)

            # Cosine similarity at masked positions
            pred_norm = F.normalize(predicted_repr, dim=-1)
            tgt_norm = F.normalize(target_repr, dim=-1)
            cos_sim = (pred_norm * tgt_norm).sum(dim=-1)
            mean_cos_sim = (cos_sim * mask_float).sum() / max(n_masked, 1)

        self.global_step += 1

        metrics = {
            "loss": loss.item(),
            "cosine_sim": mean_cos_sim.item(),
            "mask_ratio_actual": mask_ratio_actual,
            "ema_decay": self._get_ema_decay(),
        }

        if self.global_step % 100 == 0:
            logger.info(
                "JEPA Step %d | Loss: %.4f | CosSim: %.4f | "
                "MaskRatio: %.2f | EMA: %.4f",
                self.global_step,
                metrics["loss"],
                metrics["cosine_sim"],
                metrics["mask_ratio_actual"],
                metrics["ema_decay"],
            )

        return metrics

    def _encode_with_encoder(
        self,
        encoder: nn.Module,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run encoder to get representations.

        Handles both cases:
            - Encoder has a direct forward that returns (batch, seq, d_model)
            - Encoder is AamDiffusionModel which needs timestep, etc.

        For AamDiffusionModel, we use timestep=0 (lowest noise level)
        to get the cleanest representation for JEPA training.

        Args:
            encoder: Encoder module.
            token_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Optional attention mask.

        Returns:
            Representations of shape (batch, seq_len, d_model).
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device

        # Try AamDiffusionModel-style forward
        try:
            # AamDiffusionModel returns (predicted, target) at a timestep
            # Use timestep 0 for the cleanest representation
            t = torch.zeros(batch_size, dtype=torch.long, device=device)
            predicted, _ = encoder(token_ids=token_ids, timestep=t)
            return predicted
        except (TypeError, AttributeError):
            pass

        # Fallback: try a generic forward
        try:
            output = encoder(token_ids)
            if isinstance(output, tuple):
                return output[0]
            if output.dim() == 3:
                return output
            raise ValueError(
                f"Unexpected encoder output shape: {output.shape}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not get representations from encoder: {e}. "
                f"Ensure encoder forward() returns (batch, seq, d_model)."
            ) from e
