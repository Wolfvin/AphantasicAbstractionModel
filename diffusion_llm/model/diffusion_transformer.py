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

This is the "brainstem" of the body — the core computation that
transforms noisy signals into coherent patterns.

Analogi: Seperti otot Jin Soun yang merespons sinyal dari otak —
model ini menerima "sinyal noise" dan "instruksi dari graph",
lalu mengubahnya menjadi gerakan yang koheren (kalimat).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_llm.config.model_config import ModelConfig


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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

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

        # Feed-forward
        self.ff_norm = AdaptiveLayerNorm(d_model, eps=norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer scales (optional, helps with deep networks)
        self.self_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cross_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.ff_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        graph_keys: Optional[torch.Tensor] = None,
        graph_values: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
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
    │    └─ AdaLN + Feed-Forward                     │
    │                                                │
    │  Output Projection: → predicted noise          │
    └────────────────────────────────────────────────┘

    Key Features:
    - Adaptive Layer Norm: timestep-conditioned normalization
    - Cross-Attention: graph conditioning guides generation
    - Layer Scales: helps training deep networks
    - RoPE: better length generalization than learned positions

    Args:
        config: ModelConfig with architecture hyperparameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input embedding (from token IDs to d_model)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Timestep embedding
        self.timestep_embedding = SinusoidalTimestepEmbedding(config.d_model)

        # Positional encoding
        if config.pos_encoding_type == "learned":
            self.position_embedding = nn.Embedding(
                config.max_seq_len, config.d_model
            )
        else:
            # RoPE is applied inside attention (no separate embedding)
            self.position_embedding = None

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
            )
            for _ in range(config.n_layers)
        ])

        # Final norm
        NormClass = nn.RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.final_norm = NormClass(config.d_model, eps=config.norm_eps)

        # Output projection (predict noise/x0/v)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        # Initialize weights
        self.apply(self._init_weights)

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
    ) -> torch.Tensor:
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

        Returns:
            Predicted noise of shape (batch, seq_len, d_model).
        """
        # Get input embeddings
        if x_t is None and token_ids is not None:
            # Create embeddings from token IDs (used for initial x_0)
            h = self.token_embedding(token_ids)
        elif x_t is not None:
            h = x_t
        else:
            raise ValueError("Either x_t or token_ids must be provided")

        # Add positional encoding
        if self.position_embedding is not None:
            seq_len = h.shape[1]
            positions = torch.arange(seq_len, device=h.device).unsqueeze(0)
            h = h + self.position_embedding(positions)

        # Embed timestep
        t_emb = self.timestep_embedding(t)

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(
                h,
                timestep_emb=t_emb,
                graph_keys=graph_keys,
                graph_values=graph_values,
            )

        # Final norm and projection
        h = self.final_norm(h)
        output = self.output_proj(h)

        return output

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
