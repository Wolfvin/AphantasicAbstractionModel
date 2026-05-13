"""AAM Diffusion LLM — Evoformer Feedback System

Adapted from Losion/AlphaFold2: iterative bidirectional feedback
at multiple architecture levels.

For AAM, the most relevant levels:
    Level 1 — Inter-Layer Recycling: Layer deep ↔ Layer shallow
    Level 2 — Bidirectional Token Update: Token old ↔ Token new
    Level 3 — Decoder ↔ Predict: Narrative output ↔ Graph conditioning
    Level 4 — Prediction → Context: Predicted narrative refines graph understanding

Core Principle: "Whenever there are two related representations, replace
one-way information flow with iterative bidirectional dialogue."

This is PERFECT for AAM's Predictive Coding:
    predict(X) → observe(Y) → belief_update(Δ)

Evoformer makes this bidirectional and iterative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EvoformerConfig:
    d_model: int = 768
    n_recycling_steps: int = 3
    dropout: float = 0.0
    use_layer_recycling: bool = True
    use_token_recycling: bool = True
    use_decoder_feedback: bool = True
    use_prediction_recycling: bool = True
    min_recycling_improvement: float = 1e-4


class LayerRecyclingBlock(nn.Module):
    """Level 1: Bidirectional feedback between deep and shallow layers."""

    def __init__(self, d_model: int, n_recycling_steps: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_recycling_steps = n_recycling_steps

        self.shallow_query_proj = nn.Linear(d_model, d_model, bias=False)
        self.deep_key_proj = nn.Linear(d_model, d_model, bias=False)
        self.deep_value_proj = nn.Linear(d_model, d_model, bias=False)
        self.revision_proj = nn.Linear(d_model, d_model, bias=False)

        self.revision_gate = nn.Sequential(
            nn.Linear(d_model * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.scale = math.sqrt(d_model)

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(hidden_states) < 2:
            return hidden_states

        n_layers = len(hidden_states)
        mid = n_layers // 2
        shallow_repr = torch.stack(hidden_states[:mid], dim=0).mean(dim=0)
        deep_repr = torch.stack(hidden_states[mid:], dim=0).mean(dim=0)

        q = self.shallow_query_proj(shallow_repr)
        k = self.deep_key_proj(deep_repr)
        v = self.deep_value_proj(deep_repr)

        k_mean = k.mean(dim=1, keepdim=True)
        v_mean = v.mean(dim=1, keepdim=True)

        scores = torch.matmul(q, k_mean.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            attn = self.dropout(attn)

        revision = torch.matmul(attn, v_mean)
        revision = self.revision_proj(revision)

        gate = self.revision_gate(torch.cat([shallow_repr, revision], dim=-1))
        revision = gate * revision

        revised = []
        for i, h in enumerate(hidden_states):
            if i < mid:
                revised.append(h + revision * (0.1 if i < mid // 2 else 0.2))
            else:
                revised.append(h + revision * 0.05)

        return revised


class BidirectionalTokenUpdate(nn.Module):
    """Level 2: Later tokens revise earlier token representations."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        self.norm = nn.RMSNorm(d_model)
        self.dropout_mod = nn.Dropout(dropout) if dropout > 0 else None
        self.scale = math.sqrt(self.d_kv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if seq_len <= 1:
            return x

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)

        if self.dropout_mod is not None:
            attn = self.dropout_mod(attn)

        backward_info = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        backward_info = self.out_proj(backward_info)

        gate = self.gate(x)
        revised = x + gate * backward_info
        revised = self.norm(revised)

        return revised


class DecoderPredictFeedback(nn.Module):
    """Level 3: Bidirectional feedback between decoder output and graph prediction.

    AAM-specific: narrative output revises graph conditioning.
    Predict v1 → Decoder refine → feedback → Update v1 → loop
    """

    def __init__(self, d_model: int, n_iterations: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_iterations = n_iterations

        self.feedback_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        self.feedback_gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        self.norm = nn.RMSNorm(d_model)
        self.dropout_mod = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, hidden_state: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
        delta = decoder_output - hidden_state
        feedback = self.feedback_proj(delta)
        gate = self.feedback_gate(hidden_state)
        feedback = gate * feedback

        if self.dropout_mod is not None:
            feedback = self.dropout_mod(feedback)

        updated = self.norm(hidden_state + feedback)
        return updated


class PredictionContextRecycling(nn.Module):
    """Level 4: Predicted narrative revises graph understanding.

    AAM-specific: the generated narrative can refine how we understand
    the graph, creating a feedback loop between output and input.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model

        self.pred_proj = nn.Linear(d_model, d_model, bias=False)
        self.context_query = nn.Linear(d_model, d_model, bias=False)
        self.pred_key = nn.Linear(d_model, d_model, bias=False)
        self.pred_value = nn.Linear(d_model, d_model, bias=False)
        self.revision_proj = nn.Linear(d_model, d_model, bias=False)
        self.revision_gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        self.norm = nn.RMSNorm(d_model)
        self.dropout_mod = nn.Dropout(dropout) if dropout > 0 else None
        self.scale = math.sqrt(d_model)

    def forward(self, hidden_states: torch.Tensor, prediction_logits: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        if prediction_logits.shape[-1] != self.d_model:
            pred_repr = self.pred_proj(prediction_logits[:, -1:, :self.d_model]
                                        if prediction_logits.dim() == 3
                                        else prediction_logits.unsqueeze(1))
        else:
            pred_repr = prediction_logits[:, -1:, :] if prediction_logits.dim() == 3 else prediction_logits.unsqueeze(1)

        q = self.context_query(hidden_states)
        k = self.pred_key(pred_repr)
        v = self.pred_value(pred_repr)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-2)

        if self.dropout_mod is not None:
            attn = self.dropout_mod(attn)

        revision = torch.matmul(attn, v)
        revision = self.revision_proj(revision)

        gate = self.revision_gate(hidden_states)
        revised = hidden_states + gate * revision
        revised = self.norm(revised)

        return revised


class EvoformerManager(nn.Module):
    """Manages Evoformer feedback levels for AAM Diffusion LLM."""

    def __init__(self, config: EvoformerConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_layer_recycling:
            self.layer_recycling = LayerRecyclingBlock(
                d_model=config.d_model,
                n_recycling_steps=config.n_recycling_steps,
                dropout=config.dropout,
            )
        else:
            self.layer_recycling = None

        if config.use_token_recycling:
            self.bidirectional_token = BidirectionalTokenUpdate(
                d_model=config.d_model,
                n_heads=max(1, config.d_model // 128),
                dropout=config.dropout,
            )
        else:
            self.bidirectional_token = None

        if config.use_decoder_feedback:
            self.decoder_feedback = DecoderPredictFeedback(
                d_model=config.d_model,
                n_iterations=config.n_recycling_steps,
                dropout=config.dropout,
            )
        else:
            self.decoder_feedback = None

        if config.use_prediction_recycling:
            self.prediction_recycling = PredictionContextRecycling(
                d_model=config.d_model,
                dropout=config.dropout,
            )
        else:
            self.prediction_recycling = None

    def recycle_layers(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.layer_recycling is not None:
            return self.layer_recycling(hidden_states)
        return hidden_states

    def bidirectional_token_update(self, x: torch.Tensor) -> torch.Tensor:
        if self.bidirectional_token is not None:
            return self.bidirectional_token(x)
        return x

    def apply_decoder_feedback(self, hidden_state: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
        if self.decoder_feedback is not None:
            return self.decoder_feedback(hidden_state, decoder_output)
        return hidden_state

    def apply_prediction_recycling(self, hidden_states: torch.Tensor, prediction_logits: torch.Tensor) -> torch.Tensor:
        if self.prediction_recycling is not None:
            return self.prediction_recycling(hidden_states, prediction_logits)
        return hidden_states

    def get_stats(self) -> Dict[str, object]:
        return {
            "level_1_layer_recycling": self.layer_recycling is not None,
            "level_2_bidirectional_token": self.bidirectional_token is not None,
            "level_3_decoder_feedback": self.decoder_feedback is not None,
            "level_4_prediction_recycling": self.prediction_recycling is not None,
            "n_recycling_steps": self.config.n_recycling_steps,
        }
