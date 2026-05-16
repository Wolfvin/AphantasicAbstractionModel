"""
AAM Diffusion LLM — Diffusion Transformer (Denoiser)

The core denoising network. Takes noisy text embeddings and graph
conditioning, and predicts the noise (or clean data) at each
diffusion timestep.

Architecture:
    Input: Noisy embeddings x_t + timestep t + graph conditioning
    Output: Predicted noise epsilon (or x_0 or v)

The transformer uses:
    - Self-attention over the text sequence
    - Cross-attention to graph conditioning (evidence, anomalies, etc.)
    - Timestep embedding (sinusoidal) injected via adaptive layer norm
    - Optional flash attention for efficiency
    - [v2.0] SwiGLU FFN (proven better in LLaMA/Mistral)
    - [v2.0] RoPE via the dedicated rope.py module
    - [v2.0] Evoformer integration points for layer recycling

This is the "brainstem" of the body — the core computation that
transforms noisy signals into coherent patterns.

Analogi: Seperti otot Jin Soun yang merespons sinyal dari otak —
model ini menerima "sinyal noise" dan "instruksi dari graph",
lalu mengubahnya menjadi gerakan yang koheren (kalimat).
"""

from __future__ import annotations

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_llm.config.model_config import ModelConfig, EvoformerConfig, MatryoshkaConfig
from diffusion_llm.model.rope import RotaryPositionEncoding


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps.

    Maps integer timesteps to d_model-dimensional vectors using
    sinusoidal position encoding, similar to Transformers.

    This allows the model to know "how noisy" the current input is,
    which is essential for the denoising process.
    """

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        # Two-layer MLP to project sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timestep indices of shape (batch,).

        Returns:
            Timestep embeddings of shape (batch, d_model).
        """
        device = t.device
        half_dim = self.d_model // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))

        return self.mlp(emb)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep.

    Instead of fixed scale/shift parameters, this layer norm
    uses the timestep embedding to produce scale and shift:

        y = (1 + scale(t)) * norm(x) + shift(t)

    This allows the model to behave differently at different
    noise levels — more "creative" at high noise, more
    "precise" at low noise.

    Analogi: Jin Soun menyesuaikan intensitas pikirannya
    berdasarkan seberapa kabur situasinya — semakin kabur,
    semakin "kreatif" pendekatannya.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.scale_proj = nn.Linear(d_model, d_model)
        self.shift_proj = nn.Linear(d_model, d_model)

        # Initialize shift to zero, scale to one
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)
        nn.init.ones_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive layer norm.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            timestep_emb: Timestep embedding of shape (batch, d_model).

        Returns:
            Normalized and modulated tensor.
        """
        normalized = self.norm(x)
        scale = (1 + self.scale_proj(timestep_emb)).unsqueeze(1)
        shift = self.shift_proj(timestep_emb).unsqueeze(1)
        return normalized * scale + shift


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention, cross-attention, and FFN.

    The block structure:
    1. Adaptive Layer Norm + Self-Attention
    2. Adaptive Layer Norm + Cross-Attention (to graph conditioning)
    3. Adaptive Layer Norm + Feed-Forward Network

    Each sub-layer has a residual connection.

    v2.0 Changes:
    - SwiGLU FFN replaces GELU FFN (proven better in LLaMA/Mistral)
    - Optional Matryoshka elastic inference on the FFN
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        norm_type: str = "rmsnorm",
        use_flash_attention: bool = True,
        use_swiglu: bool = True,
        matryoshka_config: Optional[MatryoshkaConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_swiglu = use_swiglu
        self.matryoshka_config = matryoshka_config

        # Norms
        NormClass = nn.RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm

        # Self-attention
        self.self_attn_norm = AdaptiveLayerNorm(d_model, eps=norm_eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention (to graph conditioning)
        self.cross_attn_norm = AdaptiveLayerNorm(d_model, eps=norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_model,
            vdim=d_model,
        )
        self.cross_attn_dropout = nn.Dropout(dropout)

        # Feed-forward — SwiGLU or legacy GELU
        self.ff_norm = AdaptiveLayerNorm(d_model, eps=norm_eps)
        if use_swiglu:
            # SwiGLU FFN (proven better in LLaMA/Mistral)
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)
            self.ff_dropout = nn.Dropout(dropout)
        else:
            # Legacy GELU FFN (backward compatible)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        # Matryoshka elastic inference (optional)
        if matryoshka_config is not None and use_swiglu:
            self._matryoshka_d_ff = d_ff
            self._matryoshka_factors = sorted(matryoshka_config.granularity_factors)
            if matryoshka_config.use_adaptive:
                self.size_selector = nn.Sequential(
                    nn.Linear(d_model, d_model // 8, bias=False),
                    nn.SiLU(),
                    nn.Linear(d_model // 8, 1, bias=False),
                    nn.Sigmoid(),
                )
        else:
            self._matryoshka_d_ff = None
            self._matryoshka_factors = None

        # Layer scales (optional, helps with deep networks)
        self.self_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cross_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.ff_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _select_matryoshka_factor(self, x: torch.Tensor) -> float:
        """Adaptive factor selection for Matryoshka inference."""
        if not hasattr(self, "size_selector"):
            return 1.0
        score = self.size_selector(x.mean(dim=1, keepdim=False))
        score_val = score.mean().item()
        min_dist = float("inf")
        best_factor = self._matryoshka_factors[-1]
        for f in self._matryoshka_factors:
            dist = abs(score_val - f)
            if dist < min_dist:
                min_dist = dist
                best_factor = f
        return best_factor

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        graph_keys: Optional[torch.Tensor] = None,
        graph_values: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        granularity_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence of shape (batch, seq_len, d_model).
            timestep_emb: Timestep embedding of shape (batch, d_model).
            graph_keys: Graph conditioning keys for cross-attention,
                shape (batch, n_graph_nodes, d_model).
            graph_values: Graph conditioning values for cross-attention,
                shape (batch, n_graph_nodes, d_model).
            causal_mask: Optional causal mask for self-attention.
            granularity_factor: Optional Matryoshka granularity factor
                for elastic inference (1.0 = full size).

        Returns:
            Output sequence of shape (batch, seq_len, d_model).
        """
        # 1. Self-attention with adaptive layer norm
        normed = self.self_attn_norm(x, timestep_emb)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + self.self_attn_scale * self.self_attn_dropout(attn_out)

        # 2. Cross-attention to graph conditioning (if available)
        if graph_keys is not None and graph_values is not None:
            normed = self.cross_attn_norm(x, timestep_emb)
            cross_out, _ = self.cross_attn(
                normed, graph_keys, graph_values,
                need_weights=False,
            )
            x = x + self.cross_attn_scale * self.cross_attn_dropout(cross_out)

        # 3. Feed-forward with adaptive layer norm
        normed = self.ff_norm(x, timestep_emb)

        if self.use_swiglu:
            # Determine Matryoshka factor
            factor = granularity_factor
            if factor is None and self._matryoshka_factors is not None:
                factor = self._select_matryoshka_factor(normed)
            elif factor is None:
                factor = 1.0

            # Clamp factor
            if self._matryoshka_factors is not None:
                factor = min(max(factor, min(self._matryoshka_factors)), 1.0)
            else:
                factor = 1.0

            d_ff_active = max(1, int(self._matryoshka_d_ff * factor)) if self._matryoshka_d_ff else self.gate_proj.out_features

            if factor >= 1.0 or self._matryoshka_d_ff is None:
                # Full-size SwiGLU
                gate = F.silu(self.gate_proj(normed))
                up = self.up_proj(normed)
                ff_out = self.down_proj(gate * up)
            else:
                # Matryoshka partial SwiGLU
                d_ff_active = max(1, int(self._matryoshka_d_ff * factor))
                gate = F.silu(F.linear(normed, self.gate_proj.weight[:d_ff_active, :]))
                up = F.linear(normed, self.up_proj.weight[:d_ff_active, :])
                ff_out = F.linear(gate * up, self.down_proj.weight[:, :d_ff_active])

            ff_out = self.ff_dropout(ff_out)
        else:
            # Legacy GELU FFN
            ff_out = self.ff(normed)

        x = x + self.ff_scale * ff_out

        return x


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer — the core denoising network for AAM.

    This transformer takes:
    - Noisy text embeddings (x_t)
    - Diffusion timestep (t)
    - Graph conditioning (evidence, anomalies, reasoning chains)

    And predicts the noise that was added (or the clean data,
    depending on prediction_type).

    Architecture Overview:
    ┌────────────────────────────────────────────────┐
    │  Input Embedding: x_t (noisy) → embedding     │
    │  + Positional Encoding (RoPE or learned)       │
    │                                                │
    │  N x TransformerBlock:                         │
    │    ├─ AdaLN + Self-Attention                   │
    │    ├─ AdaLN + Cross-Attention (to graph)       │
    │    └─ AdaLN + SwiGLU FFN (Matryoshka)          │
    │                                                │
    │  [Evoformer Layer Recycling — optional]        │
    │                                                │
    │  Output Projection: → predicted noise          │
    └────────────────────────────────────────────────┘

    Key Features:
    - Adaptive Layer Norm: timestep-conditioned normalization
    - Cross-Attention: graph conditioning guides generation
    - Layer Scales: helps training deep networks
    - RoPE: better length generalization than learned positions
    - [v2.0] SwiGLU FFN: proven better than GELU in LLaMA/Mistral
    - [v2.0] Matryoshka: elastic inference at multiple sizes
    - [v2.0] Evoformer: layer recycling for iterative refinement

    Args:
        config: ModelConfig with architecture hyperparameters.
        evoformer_config: Optional EvoformerConfig for layer recycling.
        matryoshka_config: Optional MatryoshkaConfig for elastic inference.
        use_swiglu: Whether to use SwiGLU FFN (default True for v2.0).
    """

    def __init__(
        self,
        config: ModelConfig,
        evoformer_config: Optional[EvoformerConfig] = None,
        matryoshka_config: Optional[MatryoshkaConfig] = None,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.config = config
        self.evoformer_config = evoformer_config
        self.matryoshka_config = matryoshka_config
        self.use_swiglu = use_swiglu

        # Input embedding (from token IDs to d_model)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Timestep embedding
        self.timestep_embedding = SinusoidalTimestepEmbedding(config.d_model)

        # Positional encoding
        if config.pos_encoding_type == "learned":
            self.position_embedding = nn.Embedding(
                config.max_seq_len, config.d_model
            )
            self.rope = None
        else:
            # RoPE is applied inside attention (no separate embedding)
            self.position_embedding = None
            # v2.0: Create RotaryPositionEncoding module for explicit RoPE
            self.rope = RotaryPositionEncoding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
            )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm_eps=config.norm_eps,
                norm_type=config.norm_type,
                use_flash_attention=config.use_flash_attention,
                use_swiglu=use_swiglu,
                matryoshka_config=matryoshka_config,
            )
            for _ in range(config.n_layers)
        ])

        # Final norm
        NormClass = nn.RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.final_norm = NormClass(config.d_model, eps=config.norm_eps)

        # Output projection (predict noise/x0/v)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        # Evoformer integration — lazy import to avoid circular deps
        self._evoformer_manager = None
        if evoformer_config is not None:
            self._init_evoformer(evoformer_config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_evoformer(self, evoformer_config: EvoformerConfig) -> None:
        """Initialize the Evoformer manager for layer recycling."""
        from diffusion_llm.model.evoformer import EvoformerManager
        self._evoformer_manager = EvoformerManager(evoformer_config)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with Xavier/GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        graph_keys: Optional[torch.Tensor] = None,
        graph_values: Optional[torch.Tensor] = None,
        granularity_factor: Optional[float] = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass: predict noise given noisy input and timestep.

        Args:
            x_t: Noisy text embeddings of shape (batch, seq_len, d_model).
                If None, token_ids must be provided.
            t: Timestep indices of shape (batch,).
            token_ids: Token IDs of shape (batch, seq_len).
                Used to create embeddings if x_t is not provided directly.
                In training, x_t comes from the noise scheduler.
            graph_keys: Graph conditioning keys for cross-attention,
                shape (batch, n_graph_nodes, d_model).
            graph_values: Graph conditioning values for cross-attention,
                shape (batch, n_graph_nodes, d_model).
            granularity_factor: Optional Matryoshka granularity factor
                for elastic inference (1.0 = full size).
            return_hidden_states: If True, also return per-layer hidden
                states for Evoformer layer recycling.

        Returns:
            Predicted noise of shape (batch, seq_len, d_model).
            If return_hidden_states is True, also returns a list of
            per-layer hidden state tensors.
        """
        # Get input embeddings
        if x_t is None and token_ids is not None:
            # Create embeddings from token IDs (used for initial x_0)
            h = self.token_embedding(token_ids)
        elif x_t is not None:
            h = x_t
        else:
            raise ValueError("Either x_t or token_ids must be provided")

        # Add positional encoding (learned positions only; RoPE is applied in attention)
        if self.position_embedding is not None:
            seq_len = h.shape[1]
            positions = torch.arange(seq_len, device=h.device).unsqueeze(0)
            h = h + self.position_embedding(positions)

        # Embed timestep
        t_emb = self.timestep_embedding(t)

        # Pass through transformer blocks, collecting hidden states for Evoformer
        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            h = block(
                h,
                timestep_emb=t_emb,
                graph_keys=graph_keys,
                graph_values=graph_values,
                granularity_factor=granularity_factor,
            )
            if return_hidden_states or self._evoformer_manager is not None:
                hidden_states.append(h)

        # Evoformer layer recycling (if enabled)
        if self._evoformer_manager is not None and len(hidden_states) > 1:
            hidden_states = self._evoformer_manager.recycle_layers(hidden_states)
            # Use the last revised hidden state as the output
            h = hidden_states[-1]

        # Final norm and projection
        h = self.final_norm(h)
        output = self.output_proj(h)

        if return_hidden_states:
            return output, hidden_states

        return output

    def apply_evoformer_token_update(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Evoformer bidirectional token update (Level 2).

        Can be called externally as part of an Evoformer recycling loop.

        Args:
            x: Hidden state tensor of shape (batch, seq_len, d_model).

        Returns:
            Revised hidden state tensor.
        """
        if self._evoformer_manager is not None:
            return self._evoformer_manager.bidirectional_token_update(x)
        return x

    def apply_evoformer_decoder_feedback(
        self,
        hidden_state: torch.Tensor,
        decoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Evoformer decoder-predict feedback (Level 3).

        Can be called externally during anchored decoder refinement.

        Args:
            hidden_state: Hidden state tensor of shape (batch, seq_len, d_model).
            decoder_output: Decoder output tensor of shape (batch, seq_len, d_model).

        Returns:
            Revised hidden state tensor.
        """
        if self._evoformer_manager is not None:
            return self._evoformer_manager.apply_decoder_feedback(hidden_state, decoder_output)
        return hidden_state

    def apply_evoformer_prediction_recycling(
        self,
        hidden_states: torch.Tensor,
        prediction_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Evoformer prediction-context recycling (Level 4).

        Can be called externally to refine graph understanding from
        predicted output.

        Args:
            hidden_states: Hidden states of shape (batch, seq_len, d_model).
            prediction_logits: Prediction logits of shape (batch, seq_len, d_model).

        Returns:
            Revised hidden state tensor.
        """
        if self._evoformer_manager is not None:
            return self._evoformer_manager.apply_prediction_recycling(hidden_states, prediction_logits)
        return hidden_states

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
