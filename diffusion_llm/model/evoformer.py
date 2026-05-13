"""AAM Diffusion LLM — Evoformer Feedback System

Adapted from Losion/AlphaFold2: iterative bidirectional feedback
at multiple architecture levels.

For AAM, the most relevant levels:
    Level 1 — Inter-Layer Recycling: Layer deep ↔ Layer shallow
    Level 2 — Bidirectional Token Update: Token old ↔ Token new
    Level 3 — Decoder ↔ Predict: Narrative output ↔ Graph conditioning
    Level 4 — Prediction → Context: Predicted narrative refines graph understanding
    Level 5 — Router-Expert Co-evolution: Graph node ↔ Sentence arrangement

Core Principle: "Whenever there are two related representations, replace
one-way information flow with iterative bidirectional dialogue."

This is PERFECT for AAM's Predictive Coding:
    predict(X) → observe(Y) → belief_update(Δ)

Evoformer makes this bidirectional and iterative.

Level 5 (RouterExpertCoevolve) — AAM-specific adaptation:
    In Losion, this handles router ↔ MoE expert co-evolution.
    For AAM, this handles: graph node ↔ sentence arrangement co-evolution.
    The co-evolve state captures the "negotiation" between graph
    understanding and narrative output — each side adjusts based on
    the other's current state, creating an iterative dialogue where
    better graph understanding leads to better narrative, and better
    narrative feedback refines graph understanding.
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
    """Configuration for Evoformer Feedback System.

    Attributes:
        d_model: Model hidden dimension.
        n_recycling_steps: Number of recycling iterations.
        dropout: Dropout rate for all sub-modules.
        use_layer_recycling: Enable Level 1 (inter-layer recycling).
        use_token_recycling: Enable Level 2 (bidirectional token update).
        use_decoder_feedback: Enable Level 3 (decoder-predict feedback).
        use_prediction_recycling: Enable Level 4 (prediction-context recycling).
        use_router_coevolve: Enable Level 5 (router-expert co-evolution).
        d_pair: Pair representation dimension for co-evolution state.
            0 means use d_model.
        min_recycling_improvement: Minimum improvement threshold for recycling.
    """

    d_model: int = 768
    n_recycling_steps: int = 3
    dropout: float = 0.0
    use_layer_recycling: bool = True
    use_token_recycling: bool = True
    use_decoder_feedback: bool = True
    use_prediction_recycling: bool = True
    use_router_coevolve: bool = True
    d_pair: int = 0  # 0 = use d_model
    min_recycling_improvement: float = 1e-4


class LayerRecyclingBlock(nn.Module):
    """Level 1: Bidirectional feedback between deep and shallow layers.

    Losion v1.9.0 gradient-flow fix: deep layers also receive a small
    revision residual (0.05 multiplier) so that ``recycled[-1]`` carries
    gradient through the revision path back to all layer_recycling
    parameters.  Without this, deep layers get no revision and the
    gradient from the final output cannot flow back through the
    revision path.
    """

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

        # Losion v1.9.0: deep-layer revision multiplier (small but nonzero
        # to maintain gradient flow through the revision path).
        self.deep_revision_multiplier: float = 0.05

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
                # Losion v1.9.0 fix: deep layers receive a small revision
                # residual so gradient flows from recycled[-1] back through
                # the revision path to all layer_recycling parameters.
                revised.append(h + revision * self.deep_revision_multiplier)

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


class RouterExpertCoevolve(nn.Module):
    """Level 5: Graph node ↔ sentence arrangement co-evolution.

    Adapted from Losion's RouterExpertCoevolve (router ↔ MoE expert
    co-evolution).  In Losion, the router distributes tokens to MoE
    experts, and expert outputs refine the router's decisions — a
    bidirectional negotiation.

    For AAM, the co-evolution is between:
        - Graph nodes: evidence from RSVS graph (the "router" side —
          which evidence to attend to)
        - Sentence arrangement: narrative output (the "expert" side —
          how to express the evidence in natural language)

    The co-evolve state captures the "negotiation" between graph
    understanding and narrative output: each side adjusts based on
    the other's current state, creating an iterative dialogue where
    better graph understanding leads to better narrative, and better
    narrative feedback refines graph understanding.

    Key design (from Losion v1.9.0):
        - ``update_state()`` returns a **differentiable** tensor so
          gradient flows through the revision path to all
          RouterExpertCoevolve parameters.
        - The internal buffer is updated with **detached** values to
          prevent unbounded gradient accumulation across training steps.

    Args:
        d_model: Model hidden dimension.
        d_pair: Pair (co-evolution state) dimension.  0 means use d_model.
        n_experts: Number of routing experts (graph attention heads).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_pair: int = 0,
        n_experts: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair if d_pair > 0 else d_model
        self.n_experts = n_experts

        # ── Graph (router) side — projects graph representations ──
        self.graph_router = nn.Linear(d_model, n_experts, bias=False)
        self.graph_adjust_proj = nn.Linear(d_model, self.d_pair, bias=False)

        # ── Narrative (expert) side — projects narrative representations ──
        self.narrative_adjust_proj = nn.Linear(d_model, self.d_pair, bias=False)

        # ── Co-evolution gate: learns how much each side influences ──
        # the negotiation state
        self.coevolve_gate = nn.Sequential(
            nn.Linear(self.d_pair * 2, self.d_pair, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_pair, self.d_pair, bias=False),
            nn.Sigmoid(),
        )

        # ── Output projections back to d_model ──
        self.graph_out_proj = nn.Linear(self.d_pair, d_model, bias=False)
        self.narrative_out_proj = nn.Linear(self.d_pair, d_model, bias=False)

        # ── Normalization ──
        self.norm_graph = nn.RMSNorm(d_model)
        self.norm_narrative = nn.RMSNorm(d_model)

        self.dropout_mod = nn.Dropout(dropout) if dropout > 0 else None

        # ── Buffers (detached from computation graph) ──
        # Co-evolve state: the shared negotiation state between
        # graph understanding and narrative output.
        self.register_buffer("coevolve_state", torch.zeros(1, 1, self.d_pair))

        # Routing adjustment: influences which graph nodes (evidence)
        # receive more attention — the graph-side "opinion".
        self.register_buffer("routing_adjustment", torch.zeros(1, self.n_experts))

    def get_routing_adjustment(self) -> torch.Tensor:
        """Return routing adjustment based on current co-evolve state.

        The adjustment influences which graph nodes (evidence) receive
        more attention — it is the graph-side "opinion" derived from
        the current negotiation state between graph understanding and
        narrative output.

        Returns:
            Tensor of shape ``(1, n_experts)`` with routing adjustments.
        """
        # Compute fresh adjustment from the current co-evolve state
        state_flat = self.coevolve_state.squeeze(1)  # (1, d_pair)
        adj = self.graph_router(state_flat)  # (1, n_experts)
        return adj + self.routing_adjustment

    def update_state(
        self,
        graph_repr: torch.Tensor,
        narrative_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Update co-evolve state; return differentiable tensor for gradient flow.

        Losion v1.9.0 pattern: the returned tensor is differentiable,
        so gradient flows back through the revision path to all
        RouterExpertCoevolve parameters.  However, the buffer is
        updated with detached values to prevent unbounded gradient
        accumulation across training steps.

        This captures the "negotiation" between:
        - Graph understanding: which evidence nodes are most relevant
        - Narrative output: how the evidence is being expressed

        Each side adjusts the co-evolve state based on its current
        representation, and the gate learns the optimal balance.

        Args:
            graph_repr: Graph node representations ``(B, S_g, d_model)``.
                Evidence from RSVS graph.
            narrative_repr: Narrative representations ``(B, S_n, d_model)``.
                Sentence arrangement output.

        Returns:
            Differentiable co-evolve state of shape ``(B, 1, d_pair)``.
        """
        # Project both sides into the co-evolution space
        g_adj = self.graph_adjust_proj(graph_repr)       # (B, S_g, d_pair)
        n_adj = self.narrative_adjust_proj(narrative_repr)  # (B, S_n, d_pair)

        # Aggregate across sequence dimension (mean pooling)
        g_pool = g_adj.mean(dim=1, keepdim=True)   # (B, 1, d_pair)
        n_pool = n_adj.mean(dim=1, keepdim=True)   # (B, 1, d_pair)

        # Co-evolution gate: learns the negotiation balance between
        # graph understanding and narrative output
        combined = torch.cat([g_pool, n_pool], dim=-1)  # (B, 1, d_pair*2)
        gate = self.coevolve_gate(combined)               # (B, 1, d_pair)

        # New state = gated negotiation between graph and narrative,
        # blended with the previous state for stability
        new_state = gate * (g_pool + n_pool) + (1.0 - gate) * self.coevolve_state

        # IMPORTANT (Losion v1.9.0): Return differentiable version so
        # gradient flows through new_state back to all
        # RouterExpertCoevolve parameters.
        differentiable_state = new_state

        # Update buffer detached — prevents cross-step gradient
        # accumulation while keeping the state current for the next
        # forward pass.
        with torch.no_grad():
            self.coevolve_state.copy_(new_state.detach())
            # Also update routing adjustment based on new state
            adj = self.graph_router(new_state.squeeze(1))  # (B, n_experts)
            self.routing_adjustment.copy_(adj.detach().mean(dim=0, keepdim=True))

        return differentiable_state

    def forward(
        self,
        graph_repr: torch.Tensor,
        narrative_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Co-evolve graph and narrative representations.

        This is the main entry point.  It updates the co-evolve state
        (capturing the negotiation between graph understanding and
        narrative output) and applies the resulting adjustments to
        both representations.

        The co-evolution works as follows:
        1. Graph and narrative representations are projected into a
           shared co-evolution space.
        2. A gated negotiation combines both perspectives.
        3. The resulting state adjusts both graph understanding
           (which evidence to attend to) and narrative output
           (how to express the evidence).

        Args:
            graph_repr: Graph node representations ``(B, S_g, d_model)``.
                Evidence from RSVS graph.
            narrative_repr: Narrative representations ``(B, S_n, d_model)``.
                Sentence arrangement output.

        Returns:
            Tuple of ``(updated_graph, updated_narrative)`` — both
            revised through the co-evolution negotiation.
        """
        # Step 1: Update co-evolve state, get differentiable state
        # (gradient flows through this to all RouterExpertCoevolve params)
        coevolve = self.update_state(graph_repr, narrative_repr)  # (B, 1, d_pair)

        # Step 2: Expand to match input sequence lengths
        coevolve_graph = coevolve.expand(-1, graph_repr.shape[1], -1)          # (B, S_g, d_pair)
        coevolve_narrative = coevolve.expand(-1, narrative_repr.shape[1], -1)  # (B, S_n, d_pair)

        # Step 3: Project back to d_model
        graph_adj = self.graph_out_proj(coevolve_graph)            # (B, S_g, d_model)
        narrative_adj = self.narrative_out_proj(coevolve_narrative)  # (B, S_n, d_model)

        # Step 4: Apply dropout
        if self.dropout_mod is not None:
            graph_adj = self.dropout_mod(graph_adj)
            narrative_adj = self.dropout_mod(narrative_adj)

        # Step 5: Residual connection + normalization
        updated_graph = self.norm_graph(graph_repr + graph_adj)
        updated_narrative = self.norm_narrative(narrative_repr + narrative_adj)

        return updated_graph, updated_narrative


class EvoformerManager(nn.Module):
    """Manages Evoformer feedback levels for AAM Diffusion LLM.

    Levels:
        1. LayerRecyclingBlock — inter-layer bidirectional feedback
        2. BidirectionalTokenUpdate — token-level bidirectional update
        3. DecoderPredictFeedback — decoder ↔ graph prediction feedback
        4. PredictionContextRecycling — prediction → context recycling
        5. RouterExpertCoevolve — graph node ↔ sentence arrangement co-evolution
    """

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

        if config.use_router_coevolve:
            self.router_coevolve = RouterExpertCoevolve(
                d_model=config.d_model,
                d_pair=config.d_pair,
                n_experts=max(1, config.d_model // 192),
                dropout=config.dropout,
            )
        else:
            self.router_coevolve = None

    # ================================================================
    # Level 1 — Layer Recycling
    # ================================================================

    def recycle_layers(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply Level 1: inter-layer recycling."""
        if self.layer_recycling is not None:
            return self.layer_recycling(hidden_states)
        return hidden_states

    # ================================================================
    # Level 2 — Bidirectional Token Update
    # ================================================================

    def bidirectional_token_update(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Level 2: bidirectional token update."""
        if self.bidirectional_token is not None:
            return self.bidirectional_token(x)
        return x

    # ================================================================
    # Level 3 — Decoder ↔ Predict Feedback
    # ================================================================

    def apply_decoder_feedback(self, hidden_state: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
        """Apply Level 3: decoder-predict feedback.

        AAM-specific: narrative output revises graph conditioning.
        """
        if self.decoder_feedback is not None:
            return self.decoder_feedback(hidden_state, decoder_output)
        return hidden_state

    def decoder_predict_feedback(self, hidden_state: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
        """Convenience method for Level 3 (self-referential alias).

        Same as :meth:`apply_decoder_feedback` — provided for
        discoverability and symmetry with the module name.
        """
        return self.apply_decoder_feedback(hidden_state, decoder_output)

    # ================================================================
    # Level 4 — Prediction → Context Recycling
    # ================================================================

    def apply_prediction_recycling(self, hidden_states: torch.Tensor, prediction_logits: torch.Tensor) -> torch.Tensor:
        """Apply Level 4: prediction-context recycling.

        AAM-specific: predicted narrative refines graph understanding.
        """
        if self.prediction_recycling is not None:
            return self.prediction_recycling(hidden_states, prediction_logits)
        return hidden_states

    def prediction_context_recycling(self, hidden_states: torch.Tensor, prediction_logits: torch.Tensor) -> torch.Tensor:
        """Convenience method for Level 4 (self-referential alias).

        Same as :meth:`apply_prediction_recycling` — provided for
        discoverability and symmetry with the module name.
        """
        return self.apply_prediction_recycling(hidden_states, prediction_logits)

    # ================================================================
    # Level 5 — Router-Expert Co-evolution
    # ================================================================

    def apply_router_coevolve(
        self,
        graph_repr: torch.Tensor,
        narrative_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Level 5: graph node ↔ sentence arrangement co-evolution.

        AAM-specific: graph understanding and narrative output negotiate
        through the co-evolve state, each adjusting based on the other.

        Args:
            graph_repr: Graph node representations ``(B, S_g, d_model)``.
            narrative_repr: Narrative representations ``(B, S_n, d_model)``.

        Returns:
            Tuple of ``(updated_graph, updated_narrative)``.
        """
        if self.router_coevolve is not None:
            return self.router_coevolve(graph_repr, narrative_repr)
        return graph_repr, narrative_repr

    def router_expert_coevolve(
        self,
        graph_repr: torch.Tensor,
        narrative_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for Level 5 (self-referential alias).

        Same as :meth:`apply_router_coevolve` — named after the
        Losion module for discoverability.

        Args:
            graph_repr: Graph node representations ``(B, S_g, d_model)``.
            narrative_repr: Narrative representations ``(B, S_n, d_model)``.

        Returns:
            Tuple of ``(updated_graph, updated_narrative)``.
        """
        return self.apply_router_coevolve(graph_repr, narrative_repr)

    # ================================================================
    # Reset
    # ================================================================

    def reset(self) -> None:
        """Reset all mutable state (buffers, counters).

        Call this at the start of a new sequence or inference run to
        clear the co-evolve state and routing adjustments from
        previous inputs.
        """
        if self.router_coevolve is not None:
            self.router_coevolve.coevolve_state.zero_()
            self.router_coevolve.routing_adjustment.zero_()

    # ================================================================
    # Statistics
    # ================================================================

    def get_stats(self) -> Dict[str, object]:
        """Return activation status for all Evoformer levels."""
        return {
            "level_1_layer_recycling": self.layer_recycling is not None,
            "level_2_bidirectional_token": self.bidirectional_token is not None,
            "level_3_decoder_feedback": self.decoder_feedback is not None,
            "level_4_prediction_recycling": self.prediction_recycling is not None,
            "level_5_router_coevolve": self.router_coevolve is not None,
            "n_recycling_steps": self.config.n_recycling_steps,
            "d_pair": self.config.d_pair if self.config.d_pair > 0 else self.config.d_model,
        }
