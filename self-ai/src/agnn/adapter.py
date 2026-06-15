"""AGNN Adapter — plug into any HuggingFace transformer.

This module provides a portable adapter that automatically detects a
model's configuration (hidden_size, num_layers, etc.) and wires up
the AGNN message passing pipeline accordingly. The goal is zero-config
integration: given any HuggingFace AutoModelForCausalLM, the adapter
should be able to:
    1. Read model.config to determine hidden_size, num_layers, etc.
    2. Initialize AGNN components with the correct dimensions.
    3. Hook into the model's forward pass to inject aggregated graph
       information into the appropriate hidden states.
    4. Provide a clean interface for the rest of the system.

This replaces the previous UnconsciousInjector approach. Instead of
a hardcoded projection from bge-m3 space, the adapter dynamically
configures itself based on the target model's architecture.

Supported model families:
    - Qwen2/Qwen3 (model_type="qwen2") — detected via config.model_type
    - Llama/Llama2/Llama3 (model_type="llama") — detected via config.model_type
    - Mistral (model_type="mistral") — detected via config.model_type
    - Gemma/Gemma2 (model_type="gemma2") — detected via config.model_type
    - Fallback: any model with config.hidden_size and config.num_hidden_layers

Detection priority:
    1. config.model_type (primary — set by HuggingFace AutoConfig)
    2. Model class name (secondary — e.g., "Qwen2ForCausalLM")
    3. Generic fallback if hidden_size + num_hidden_layers are present

No hardcoded hidden_size or num_layers values exist in this module —
everything is read from model.config at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
#  Custom Exception
# ──────────────────────────────────────────────────────

class UnsupportedModelError(Exception):
    """Raised when the adapter cannot determine a model's architecture.

    This happens when:
      - model.config has no model_type field AND the class name is unrecognizable
      - model.config lacks hidden_size or num_hidden_layers
      - The model is None and no config is provided

    The error message should clearly state what is missing so the caller
    can decide how to proceed (e.g., manually provide a ModelProfile).
    """
    pass


# ──────────────────────────────────────────────────────
#  Model Architecture Mapping
# ──────────────────────────────────────────────────────

# Maps config.model_type → architecture family name used in ModelProfile
_MODEL_TYPE_TO_ARCH = {
    "qwen2": "qwen",
    "qwen3": "qwen",
    "llama": "llama",
    "mistral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma",
}

# Maps class name substrings → architecture family name
# Used as fallback when config.model_type is absent
_CLASS_NAME_TO_ARCH = {
    "Qwen2ForCausalLM": "qwen",
    "Qwen3ForCausalLM": "qwen",
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma",
}


# ──────────────────────────────────────────────────────
#  ModelProfile
# ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelProfile:
    """Immutable profile of a transformer model's architecture.

    Captures the essential dimensions needed by AGNN to correctly
    wire up graph operations, hook injection, and projection layers.

    Attributes:
        model_id: Human-readable identifier (e.g., "Qwen/Qwen3-0.6B").
            Typically from model.config._name_or_path or passed explicitly.
        hidden_size: Dimensionality of the model's hidden states.
            Corresponds to config.hidden_size.
        num_layers: Number of transformer layers.
            Corresponds to config.num_hidden_layers.
        recommended_hook_layer: The layer index to attach forward hooks.
            Defaults to num_layers // 2 (middle layer), which research
            shows captures the richest semantic information.
        architecture: Architecture family name: "qwen", "llama", "mistral",
            "gemma", or "unknown". Determines hook placement strategy and
            any architecture-specific quirks.
    """

    model_id: str
    hidden_size: int
    num_layers: int
    recommended_hook_layer: int
    architecture: str

    def __post_init__(self):
        """Validate profile fields after initialization."""
        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {self.hidden_size}"
            )
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {self.num_layers}"
            )
        if not self.recommended_hook_layer < self.num_layers:
            raise ValueError(
                f"recommended_hook_layer ({self.recommended_hook_layer}) must be "
                f"less than num_layers ({self.num_layers})"
            )


# ──────────────────────────────────────────────────────
#  SelfAdapter
# ──────────────────────────────────────────────────────

class SelfAdapter:
    """Portable adapter that auto-detects model architecture and wires AGNN.

    SelfAdapter bridges the gap between arbitrary HuggingFace transformer
    models and the AGNN pipeline. Given a model (or its config), it:

      1. Detects the model family (Qwen, Llama, Mistral, Gemma, or unknown)
      2. Extracts hidden_size, num_layers from model.config
      3. Computes the recommended hook layer (middle layer by default)
      4. Provides projection to map arbitrary vectors into the model's space
      5. Adapts AGNNGraph dimensions to match the model

    Usage with a loaded model:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        adapter = SelfAdapter.from_model(model)
        print(adapter.profile)
        # ModelProfile(model_id='Qwen/Qwen3-0.6B', hidden_size=1024,
        #              num_layers=28, recommended_hook_layer=14,
        #              architecture='qwen')

    Usage with a mock config (for testing):
        adapter = SelfAdapter.from_config(mock_config)
        adapter.adapt_graph(graph)

    Design principles:
        - Zero hardcoding: no hidden_size=1024 or num_layers=28 anywhere
        - Works without a real model loaded (mock config support)
        - Clear error messages when config is incomplete
        - Does not modify SelfCore's existing API
    """

    def __init__(self, profile: ModelProfile, config: Any = None):
        """Initialize SelfAdapter with a pre-built profile.

        Most callers should use SelfAdapter.from_model() or
        SelfAdapter.from_config() instead of calling __init__ directly.

        Args:
            profile: A fully populated ModelProfile.
            config: The original config object (optional, kept for
                reference and future use).
        """
        self._profile = profile
        self._config = config
        # Lazy-initialized projection matrix (source_dim → hidden_size)
        self._projection_matrix: Optional[np.ndarray] = None

    # ──────────────── Factory Methods ────────────────

    @classmethod
    def from_model(cls, model: Any) -> "SelfAdapter":
        """Create a SelfAdapter by auto-detecting from a loaded model.

        This is the primary factory method. It reads model.config to
        determine the architecture family, hidden_size, num_layers, etc.

        Detection strategy (in priority order):
          1. config.model_type — the canonical HuggingFace field
          2. Model class name — fallback when model_type is missing
          3. Generic — if hidden_size and num_hidden_layers exist but
             model_type is unknown, treat as "unknown" architecture

        Args:
            model: A HuggingFace transformer model with a .config attribute
                (e.g., AutoModelForCausalLM instance).

        Returns:
            SelfAdapter configured for this model.

        Raises:
            UnsupportedModelError: If the model's config cannot be read
                or lacks essential fields (hidden_size, num_hidden_layers).
        """
        if model is None:
            raise UnsupportedModelError(
                "Cannot create SelfAdapter from None model. "
                "Pass a HuggingFace model with a .config attribute."
            )

        config = getattr(model, "config", None)
        if config is None:
            raise UnsupportedModelError(
                "Model has no .config attribute. "
                "Ensure you pass a HuggingFace AutoModelForCausalLM instance."
            )

        return cls.from_config(config, model=model)

    @classmethod
    def from_config(cls, config: Any, model: Any = None) -> "SelfAdapter":
        """Create a SelfAdapter from a config object (no model needed).

        Useful for testing with mock configs, or when the model hasn't
        been loaded yet but the config is available.

        Args:
            config: A HuggingFace PretrainedConfig or compatible object
                with at least hidden_size and num_hidden_layers attributes.
            model: Optional model reference (used for class name detection
                as a fallback when config.model_type is missing).

        Returns:
            SelfAdapter configured from the given config.

        Raises:
            UnsupportedModelError: If essential config fields are missing.
        """
        # ── Extract hidden_size ──
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise UnsupportedModelError(
                "config does not have 'hidden_size' attribute. "
                "This field is required to configure AGNN dimensions. "
                "If using a custom model, provide a config with hidden_size."
            )

        # ── Extract num_layers ──
        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            raise UnsupportedModelError(
                "config does not have 'num_hidden_layers' attribute. "
                "This field is required to determine hook placement. "
                "If using a custom model, provide a config with num_hidden_layers."
            )

        # ── Extract model_id ──
        model_id = getattr(config, "_name_or_path", None) or "unknown"

        # ── Detect architecture family ──
        architecture = cls._detect_architecture(config, model)

        # ── Compute recommended hook layer (middle layer) ──
        recommended_hook_layer = num_layers // 2

        profile = ModelProfile(
            model_id=model_id,
            hidden_size=hidden_size,
            num_layers=num_layers,
            recommended_hook_layer=recommended_hook_layer,
            architecture=architecture,
        )

        logger.info(
            "SelfAdapter created: model_id=%s, hidden_size=%d, "
            "num_layers=%d, hook_layer=%d, architecture=%s",
            profile.model_id, profile.hidden_size, profile.num_layers,
            profile.recommended_hook_layer, profile.architecture,
        )

        return cls(profile=profile, config=config)

    # ──────────────── Architecture Detection ────────────────

    @staticmethod
    def _detect_architecture(config: Any, model: Any = None) -> str:
        """Detect the model architecture family from config and model.

        Priority:
          1. config.model_type → known mapping
          2. Model class name → known mapping
          3. "unknown" if both fail (but config has hidden_size/num_layers)

        Args:
            config: The model's config object.
            model: Optional model instance (for class name detection).

        Returns:
            Architecture family string: "qwen", "llama", "mistral",
            "gemma", or "unknown".
        """
        # Strategy 1: config.model_type
        model_type = getattr(config, "model_type", None)
        if model_type and model_type in _MODEL_TYPE_TO_ARCH:
            return _MODEL_TYPE_TO_ARCH[model_type]

        # Strategy 2: Model class name
        if model is not None:
            class_name = type(model).__name__
            for known_class, arch in _CLASS_NAME_TO_ARCH.items():
                if known_class in class_name:
                    return arch

        # Strategy 3: Generic fallback
        # If we got here, hidden_size and num_hidden_layers must exist
        # (caller already checked), so this is a valid but unknown arch
        return "unknown"

    # ──────────────── Public Interface ────────────────

    @property
    def profile(self) -> ModelProfile:
        """Return the model profile for this adapter."""
        return self._profile

    def get_hook_layer(self, model: Any) -> Any:
        """Return the transformer layer object for hook attachment.

        For HuggingFace causal LMs, the layers are stored in:
          - model.model.layers[hook_layer_index]

        This returns the actual nn.Module that can be passed to
        register_forward_hook().

        Args:
            model: The HuggingFace transformer model.

        Returns:
            The transformer layer module at the recommended hook index.

        Raises:
            UnsupportedModelError: If the model's layer structure is
                not recognized (no model.model.layers attribute).
        """
        hook_idx = self._profile.recommended_hook_layer

        # Try the standard HuggingFace path: model.model.layers[i]
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2 style: model.transformer.h[i]
            layers = model.transformer.h

        if layers is None:
            raise UnsupportedModelError(
                f"Cannot find transformer layers in model. "
                f"Expected model.model.layers or model.transformer.h. "
                f"Model type: {type(model).__name__}"
            )

        if hook_idx >= len(layers):
            raise UnsupportedModelError(
                f"Hook layer index {hook_idx} is out of range "
                f"(model has {len(layers)} layers). "
                f"Profile says num_layers={self._profile.num_layers}, "
                f"but actual layer count differs."
            )

        return layers[hook_idx]

    def project_to_hidden(self, vector: np.ndarray) -> np.ndarray:
        """Project an arbitrary vector into the model's hidden_size space.

        This handles the dimensionality mismatch between AGNN's internal
        embeddings (typically 64-dim) and the model's hidden states
        (e.g., 1024-dim for Qwen3, 2048-dim for Llama).

        Projection strategy:
          - If source_dim == hidden_size: return as-is (no projection needed)
          - If source_dim < hidden_size: zero-pad the right side
          - If source_dim > hidden_size: truncate to hidden_size

        For production use, a learned linear projection would be better,
        but zero-pad/truncate is sufficient for the initial integration
        and avoids introducing trainable parameters.

        Args:
            vector: Input vector of any dimensionality.

        Returns:
            Projected vector of shape (hidden_size,).
        """
        target_dim = self._profile.hidden_size
        source_dim = len(vector)

        if source_dim == target_dim:
            return vector.astype(np.float32)

        if source_dim < target_dim:
            # Zero-pad: [vector, 0, 0, ..., 0]
            result = np.zeros(target_dim, dtype=np.float32)
            result[:source_dim] = vector
            return result

        # Truncate: vector[:hidden_size]
        return vector[:target_dim].astype(np.float32)

    def adapt_graph(self, graph: Any) -> None:
        """Adapt an AGNNGraph's embedding dimension to match the model.

        Sets the graph's _embedding_dim to match the model's hidden_size,
        ensuring that graph node embeddings are compatible with the model's
        hidden state space.

        IMPORTANT: This modifies the graph in-place. If the graph already
        has nodes, their embeddings will be resized on next access (via
        add_node's existing logic). Call this BEFORE populating the graph
        for best results.

        Args:
            graph: An AGNNGraph instance to adapt.
        """
        target_dim = self._profile.hidden_size
        current_dim = graph._embedding_dim

        if current_dim == target_dim:
            logger.debug(
                "Graph embedding_dim (%d) already matches model hidden_size — "
                "no adaptation needed", current_dim
            )
            return

        logger.info(
            "Adapting graph embedding_dim: %d → %d (model: %s, arch: %s)",
            current_dim, target_dim,
            self._profile.model_id, self._profile.architecture,
        )
        graph._embedding_dim = target_dim

    def __repr__(self) -> str:
        return (
            f"SelfAdapter(model_id={self._profile.model_id!r}, "
            f"hidden_size={self._profile.hidden_size}, "
            f"num_layers={self._profile.num_layers}, "
            f"hook_layer={self._profile.recommended_hook_layer}, "
            f"architecture={self._profile.architecture!r})"
        )
