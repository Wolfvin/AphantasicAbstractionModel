"""
AAM Diffusion LLM — Graph Conditioning Encoder

Encodes structured graph data into a conditioning vector that guides
the diffusion process. This is the KEY differentiator from general LLMs:
the model is conditioned on GRAPH STRUCTURE, not just text prompts.

The graph encoder takes:
    - Evidence nodes (what the graph knows)
    - Compositions (how concepts compose)
    - Confidence scores (how sure the graph is)
    - Anomalies (what doesn't fit)
    - Reasoning chains (how the graph reached conclusions)
    - Temporal context (when events happened)

And produces a conditioning representation that the diffusion model
uses to guide denoising.

Analogi: Seperti otak Jin Soun mengirimkan sinyal ke pita suaranya —
graph memberi "tahu" apa yang harus dikatakan, dan encoder ini
menerjemahkan "pengetahuan graph" menjadi "instruksi untuk tubuh".
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_llm.config.model_config import GraphEncoderConfig


class ConfidenceEmbedding(nn.Module):
    """Embed confidence scores as continuous values.

    Maps [0, 1] confidence scores to d_graph-dimensional vectors
    using sinusoidal encoding for smooth interpolation.

    Analogi: Jin Soun tahu bedanya "aku yakin 100%" vs "mungkin 60%"
    — encoding ini mengajarkan model membedakan juga.
    """

    def __init__(self, d_graph: int):
        super().__init__()
        self.d_graph = d_graph
        # Learnable projection from scalar to d_graph
        self.projection = nn.Sequential(
            nn.Linear(1, d_graph // 4),
            nn.GELU(),
            nn.Linear(d_graph // 4, d_graph),
        )

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        """Embed confidence scores.

        Args:
            confidence: Tensor of shape (..., 1) with values in [0, 1].

        Returns:
            Tensor of shape (..., d_graph).
        """
        if confidence.dim() == 0:
            confidence = confidence.unsqueeze(0)
        if confidence.dim() == 1:
            confidence = confidence.unsqueeze(-1)
        return self.projection(confidence)


class TemporalEmbedding(nn.Module):
    """Embed temporal context as position-aware vectors.

    Uses sinusoidal positional encoding adapted for timestamps,
    allowing the model to understand time-based relationships.

    Analogi: Jin Soun mengingat bahwa "kejadian A terjadi 3 hari
    sebelum kejadian B" — temporal embedding mengajarkan model
    memahami hubungan waktu antar kejadian.
    """

    def __init__(self, d_graph: int, max_period: int = 10000):
        super().__init__()
        self.d_graph = d_graph
        self.max_period = max_period
        self.projection = nn.Sequential(
            nn.Linear(d_graph, d_graph),
            nn.GELU(),
            nn.Linear(d_graph, d_graph),
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Embed timestamps.

        Args:
            timestamps: Tensor of shape (batch, n_events) with normalized
                timestamps (0 = earliest, 1 = latest).

        Returns:
            Tensor of shape (batch, n_events, d_graph).
        """
        batch_size, n_events = timestamps.shape
        device = timestamps.device

        # Sinusoidal encoding
        half_dim = self.d_graph // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = timestamps.float().unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if emb.shape[-1] < self.d_graph:
            # Pad if d_graph is odd
            emb = F.pad(emb, (0, self.d_graph - emb.shape[-1]))

        return self.projection(emb)


class NodeEncoder(nn.Module):
    """Encode a single evidence node or composition.

    Each node is represented as:
    - Text embedding (from the tokenizer's vocabulary)
    - Confidence score
    - Optional temporal context
    - Source trust score

    These are combined into a single d_graph-dimensional vector.
    """

    def __init__(
        self,
        d_graph: int,
        vocab_size: int = 32000,
        embed_confidence: bool = True,
        embed_temporal: bool = True,
    ):
        super().__init__()
        self.d_graph = d_graph

        # Text embedding (will be shared with the main model)
        self.text_embed = nn.Embedding(vocab_size, d_graph)

        # Confidence embedding
        self.use_confidence = embed_confidence
        if embed_confidence:
            self.conf_embed = ConfidenceEmbedding(d_graph)

        # Temporal embedding
        self.use_temporal = embed_temporal
        if embed_temporal:
            self.temporal_embed = TemporalEmbedding(d_graph)

        # Fusion layer — always build for max possible inputs
        # At runtime, we may have fewer (e.g., no temporal data provided),
        # so we use a flexible approach: always concatenate all available
        # embeddings and project through a layer that handles the max size.
        self._n_max_inputs = 1 + int(embed_confidence) + int(embed_temporal)
        self.fusion = nn.Sequential(
            nn.Linear(d_graph * self._n_max_inputs, d_graph),
            nn.GELU(),
            nn.LayerNorm(d_graph),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of evidence nodes.

        Args:
            token_ids: Token IDs of shape (batch, n_nodes, seq_len).
            confidence: Confidence scores of shape (batch, n_nodes).
            timestamps: Timestamps of shape (batch, n_nodes).

        Returns:
            Encoded nodes of shape (batch, n_nodes, d_graph).
        """
        # Text embedding: mean pool over sequence length
        text_emb = self.text_embed(token_ids).mean(dim=-2)  # (batch, n_nodes, d_graph)

        embeddings = [text_emb]

        if self.use_confidence:
            if confidence is not None:
                conf_emb = self.conf_embed(confidence.unsqueeze(-1))  # (batch, n_nodes, d_graph)
                embeddings.append(conf_emb)
            else:
                # Zero-pad to maintain consistent dimension
                embeddings.append(torch.zeros_like(text_emb))

        if self.use_temporal:
            if timestamps is not None:
                temp_emb = self.temporal_embed(timestamps)  # (batch, n_nodes, d_graph)
                embeddings.append(temp_emb)
            else:
                embeddings.append(torch.zeros_like(text_emb))

        # Fuse all embeddings
        combined = torch.cat(embeddings, dim=-1)
        return self.fusion(combined)


class GraphAttentionLayer(nn.Module):
    """Multi-head attention layer for graph-structured data.

    Unlike standard self-attention, this operates on graph nodes
    where edges represent structural relationships (compositions,
    evidence links, temporal connections).

    For now, we use standard multi-head attention over the node
    sequence, as the structural information is already encoded
    in the node features. Future versions can incorporate explicit
    edge structure via graph attention networks (GAT).
    """

    def __init__(self, d_graph: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_graph,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_graph)
        self.ff = nn.Sequential(
            nn.Linear(d_graph, d_graph * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_graph * 4, d_graph),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(d_graph)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (batch, n_nodes, d_graph).
            mask: Optional attention mask.

        Returns:
            Updated node features of same shape.
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm_ff(x + ff_out)

        return x


class GraphConditioningEncoder(nn.Module):
    """Encode graph-structured conditioning data for the diffusion model.

    This encoder takes structured data from the RSVS Knowledge Graph
    and produces conditioning vectors that guide the diffusion process.

    The encoding process:
    1. Encode each evidence node (text + confidence + temporal)
    2. Encode compositions (how concepts relate)
    3. Encode anomalies (what doesn't fit)
    4. Encode reasoning chain (step-by-step logic)
    5. Aggregate via graph attention layers
    6. Project to conditioning vector for the diffusion model

    Output modes (conditioning_method):
    - 'cross_attention': Returns (K, V) pairs for cross-attention in transformer
    - 'ada_ln': Returns scale/shift parameters for adaptive layer norm
    - 'concat': Returns a conditioning prefix to concatenate with input

    Args:
        config: GraphEncoderConfig with hyperparameters.
        vocab_size: Vocabulary size (must match tokenizer).
    """

    def __init__(
        self,
        config: GraphEncoderConfig,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.config = config
        self.conditioning_method = config.conditioning_method

        # Node encoders for different graph element types
        self.evidence_encoder = NodeEncoder(
            d_graph=config.d_graph,
            vocab_size=vocab_size,
            embed_confidence=config.embed_confidence,
            embed_temporal=config.embed_temporal,
        )

        self.composition_encoder = NodeEncoder(
            d_graph=config.d_graph,
            vocab_size=vocab_size,
            embed_confidence=config.embed_confidence,
            embed_temporal=False,  # Compositions don't have temporal info
        )

        self.anomaly_encoder = NodeEncoder(
            d_graph=config.d_graph,
            vocab_size=vocab_size,
            embed_confidence=True,  # Anomalies always have confidence
            embed_temporal=config.embed_temporal,
        )

        self.reasoning_encoder = NodeEncoder(
            d_graph=config.d_graph,
            vocab_size=vocab_size,
            embed_confidence=True,  # Reasoning steps have confidence
            embed_temporal=False,
        )

        # Source trust embedding
        self.trust_embed = ConfidenceEmbedding(config.d_graph)

        # Graph attention layers for cross-node interaction
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(
                d_graph=config.d_graph,
                n_heads=config.n_graph_heads,
                dropout=0.1,
            )
            for _ in range(config.n_graph_layers)
        ])

        # Conditioning projection depends on method
        # d_model_out will be set via set_output_dim() or defaults to d_graph
        self._d_model_out = config.d_graph

        if self.conditioning_method == "cross_attention":
            # Project to (K, V) for cross-attention
            self.key_proj = nn.Linear(config.d_graph, self._d_model_out)
            self.value_proj = nn.Linear(config.d_graph, self._d_model_out)

        elif self.conditioning_method == "ada_ln":
            # Project to scale and shift for adaptive layer norm
            self.scale_proj = nn.Linear(config.d_graph, self._d_model_out)
            self.shift_proj = nn.Linear(config.d_graph, self._d_model_out)

        elif self.conditioning_method == "concat":
            # Project to a prefix sequence
            self.concat_proj = nn.Linear(config.d_graph, self._d_model_out)

        # Global pooling for summary
        self.global_pool_proj = nn.Sequential(
            nn.Linear(config.d_graph, config.d_graph),
            nn.GELU(),
            nn.Linear(config.d_graph, config.d_graph),
        )

        # Type embeddings for different graph element types
        self.type_embeddings = nn.Embedding(4, config.d_graph)
        # 0 = evidence, 1 = composition, 2 = anomaly, 3 = reasoning

    def set_output_dim(self, d_model_out: int) -> None:
        """Set the output dimension for the projection layers.

        This must be called after __init__ if d_graph != d_model
        (which is typically the case when the graph encoder's d_graph
        differs from the transformer's d_model).

        Args:
            d_model_out: Output dimension (typically the transformer's d_model).
        """
        if d_model_out == self._d_model_out:
            return  # No change needed

        self._d_model_out = d_model_out

        # Rebuild projection layers with new output dim
        if self.conditioning_method == "cross_attention":
            self.key_proj = nn.Linear(self.config.d_graph, d_model_out)
            self.value_proj = nn.Linear(self.config.d_graph, d_model_out)
        elif self.conditioning_method == "ada_ln":
            self.scale_proj = nn.Linear(self.config.d_graph, d_model_out)
            self.shift_proj = nn.Linear(self.config.d_graph, d_model_out)
        elif self.conditioning_method == "concat":
            self.concat_proj = nn.Linear(self.config.d_graph, d_model_out)

    def forward(
        self,
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
        batch_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode graph conditioning data.

        All inputs are optional — the encoder handles missing data gracefully.

        Args:
            evidence_ids: Evidence node token IDs, shape (batch, n_evidence, seq_len).
            evidence_confidence: Evidence confidence scores, shape (batch, n_evidence).
            evidence_timestamps: Evidence timestamps, shape (batch, n_evidence).
            composition_ids: Composition token IDs, shape (batch, n_compositions, seq_len).
            composition_confidence: Composition confidence, shape (batch, n_compositions).
            anomaly_ids: Anomaly token IDs, shape (batch, n_anomalies, seq_len).
            anomaly_confidence: Anomaly confidence, shape (batch, n_anomalies).
            anomaly_timestamps: Anomaly timestamps, shape (batch, n_anomalies).
            reasoning_ids: Reasoning step token IDs, shape (batch, n_steps, seq_len).
            reasoning_confidence: Reasoning confidence, shape (batch, n_steps).
            source_trust: Source trust score, shape (batch,).

        Returns:
            Dictionary with conditioning tensors depending on conditioning_method:
            - 'cross_attention': {'keys': ..., 'values': ..., 'global': ...}
            - 'ada_ln': {'scale': ..., 'shift': ..., 'global': ...}
            - 'concat': {'prefix': ..., 'global': ...}
        """
        batch_size_inferred = self._infer_batch_size(
            evidence_ids, composition_ids, anomaly_ids, reasoning_ids
        )
        device = next(self.parameters()).device

        # Encode each type of graph element
        node_embeddings = []
        type_indices = []

        # Evidence nodes
        if evidence_ids is not None:
            evidence_emb = self.evidence_encoder(
                evidence_ids, evidence_confidence, evidence_timestamps
            )
            # Add type embedding
            type_emb = self.type_embeddings(
                torch.zeros(evidence_emb.shape[1], dtype=torch.long, device=device)
            )
            evidence_emb = evidence_emb + type_emb.unsqueeze(0)
            node_embeddings.append(evidence_emb)
            type_indices.extend([0] * evidence_emb.shape[1])

        # Compositions
        if composition_ids is not None:
            comp_emb = self.composition_encoder(
                composition_ids, composition_confidence
            )
            type_emb = self.type_embeddings(
                torch.ones(comp_emb.shape[1], dtype=torch.long, device=device)
            )
            comp_emb = comp_emb + type_emb.unsqueeze(0)
            node_embeddings.append(comp_emb)
            type_indices.extend([1] * comp_emb.shape[1])

        # Anomalies
        if anomaly_ids is not None:
            anom_emb = self.anomaly_encoder(
                anomaly_ids, anomaly_confidence, anomaly_timestamps
            )
            type_emb = self.type_embeddings(
                torch.full((anom_emb.shape[1],), 2, dtype=torch.long, device=device)
            )
            anom_emb = anom_emb + type_emb.unsqueeze(0)
            node_embeddings.append(anom_emb)
            type_indices.extend([2] * anom_emb.shape[1])

        # Reasoning steps
        if reasoning_ids is not None:
            reason_emb = self.reasoning_encoder(
                reasoning_ids, reasoning_confidence
            )
            type_emb = self.type_embeddings(
                torch.full((reason_emb.shape[1],), 3, dtype=torch.long, device=device)
            )
            reason_emb = reason_emb + type_emb.unsqueeze(0)
            node_embeddings.append(reason_emb)
            type_indices.extend([3] * reason_emb.shape[1])

        # If no graph data, return zero conditioning
        if not node_embeddings:
            bsz = batch_size or batch_size_inferred
            dummy = torch.zeros(
                bsz, 1, self.config.d_graph, device=device
            )
            return self._project_conditioning(dummy)

        # Concatenate all node embeddings
        all_nodes = torch.cat(node_embeddings, dim=1)  # (batch, n_total_nodes, d_graph)

        # Add source trust as a global bias
        if source_trust is not None:
            trust_emb = self.trust_embed(source_trust.unsqueeze(-1))  # (batch, d_graph)
            # Broadcast trust to all nodes
            all_nodes = all_nodes + trust_emb.unsqueeze(1) * 0.1  # Small influence

        # Apply graph attention layers
        for layer in self.graph_layers:
            all_nodes = layer(all_nodes)

        # Compute global conditioning (mean pool)
        global_cond = all_nodes.mean(dim=1)  # (batch, d_graph)
        global_cond = self.global_pool_proj(global_cond)

        # Project based on conditioning method
        result = self._project_conditioning(all_nodes)
        result["global"] = global_cond

        return result

    def _project_conditioning(
        self, node_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project node features to conditioning format.

        Args:
            node_features: Shape (batch, n_nodes, d_graph).

        Returns:
            Dictionary with conditioning tensors.
        """
        result = {}

        if self.conditioning_method == "cross_attention":
            result["keys"] = self.key_proj(node_features)
            result["values"] = self.value_proj(node_features)

        elif self.conditioning_method == "ada_ln":
            # Use mean-pooled features for scale/shift
            pooled = node_features.mean(dim=1)
            result["scale"] = self.scale_proj(pooled)
            result["shift"] = self.shift_proj(pooled)

        elif self.conditioning_method == "concat":
            result["prefix"] = self.concat_proj(node_features)

        return result

    @staticmethod
    def _infer_batch_size(*tensors) -> int:
        """Infer batch size from the first non-None tensor."""
        for t in tensors:
            if t is not None:
                return t.shape[0]
        return 1
