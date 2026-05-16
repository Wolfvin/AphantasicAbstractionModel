"""AAM Diffusion LLM — Rotary Position Encoding (RoPE)

Implements Rotary Position Encoding from Su et al. (2021).
Better length generalization than learned positional encodings.
Applied inside attention computation, not as a separate embedding.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding (RoPE).
    
    Applies rotary embeddings to query and key tensors before
    attention computation. This allows the model to naturally
    encode relative positions through the rotation matrix.
    """

    def __init__(self, d_model: int, max_seq_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for max_seq_len
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key.
        
        Args:
            q: Query tensor (batch, n_heads, seq_len, d_head)
            k: Key tensor (batch, n_heads, seq_len, d_head)
            seq_len: Sequence length (inferred from q if None)
            offset: Position offset (for KV cache)
            
        Returns:
            Tuple (rotated_q, rotated_k)
        """
        if seq_len is None:
            seq_len = q.shape[2]

        if offset + seq_len > self.max_seq_len:
            self._precompute_cache(offset + seq_len)

        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)

        return q_rot, k_rot

    @staticmethod
    def _apply_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]

        # Handle dimension mismatch
        if cos.shape[-1] != d:
            cos = cos[..., :d]
            sin = sin[..., :d]

        cos1 = cos[..., :d // 2]
        cos2 = cos[..., d // 2:]
        sin1 = sin[..., :d // 2]
        sin2 = sin[..., d // 2:]

        rotated = torch.cat([
            x1 * cos1 - x2 * sin1,
            x1 * sin2 + x2 * cos2,
        ], dim=-1)

        return rotated


def apply_rope_to_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    d_model: int,
    seq_len: int,
    offset: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional RoPE application — use when you don't want a module.
    
    Args:
        q: Query tensor (batch, n_heads, seq_len, d_head)
        k: Key tensor (batch, n_heads, seq_len, d_head)
        d_model: Model dimension
        seq_len: Sequence length
        offset: Position offset
        device: Device
        
    Returns:
        Tuple (rotated_q, rotated_k)
    """
    d_head = q.shape[-1]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, dtype=torch.float32, device=device) / d_head))
    
    positions = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)

    d = q.shape[-1]
    x1_q, x2_q = q[..., :d // 2], q[..., d // 2:]
    x1_k, x2_k = k[..., :d // 2], k[..., d // 2:]

    cos1, cos2 = cos[..., :d // 2], cos[..., d // 2:]
    sin1, sin2 = sin[..., :d // 2], sin[..., d // 2:]

    q_rot = torch.cat([x1_q * cos1 - x2_q * sin1, x1_q * sin2 + x2_q * cos2], dim=-1)
    k_rot = torch.cat([x1_k * cos1 - x2_k * sin1, x1_k * sin2 + x2_k * cos2], dim=-1)

    return q_rot, k_rot
