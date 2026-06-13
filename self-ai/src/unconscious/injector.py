# @WHO:   self-ai/src/unconscious/injector.py
# @WHAT:  Inject experience vectors into Qwen3 hidden states during forward pass
# @PART:  self-ai/unconscious
# @ENTRY: UnconsciousInjector

"""UnconsciousInjector — inject experience vectors into Qwen3's hidden states.

Purpose:
    SELF's understanding graph stores experience as UnderstandingNodes with
    bge-m3 condition_embedding vectors (1024-dim). Currently, these only
    influence answers via the conscious path (text prompt injection).

    UnconsciousInjector adds an UNCONSCIOUS path: it averages the embedding
    vectors of relevant UnderstandingNodes into a single "experience vector",
    then injects it into Qwen3's hidden state during forward pass via a
    forward hook on a middle transformer layer.

    This is activation steering — the model "feels" the experience without
    the experience appearing in the prompt text. The injection is additive
    (not replace), applied only at the last token position (the prediction
    position), and removed after generation completes.

Architecture:
    UnderstandingNode.condition_embedding (1024-dim)
        ↓ average across retrieved nodes
    experience_vector (1024-dim float tensor)
        ↓ register_forward_hook on Qwen3 middle layer
    hidden_state[:, -1, :] += experience_vector
        ↓ model.generate() continues with steered activations
    output tokens influenced by experience

Usage:
    injector = UnconsciousInjector(qwen3_model)
    with injector.active(experience_nodes):
        output = qwen3_model.generate(input_ids, ...)

    # Check what was injected (for introspection)
    log = injector.last_injection_log

Qwen3-0.6B architecture notes:
    - 28 transformer layers (model.model.layers)
    - hidden_size = 896 (NOT 1024 like bge-m3)
    - We inject at layer 14 (middle of 28 layers)
    - A simple linear projection bridges 1024 → 896 dims
    - Projection is NOT trained — initialized with truncated SVD-like mapping

Constraint:
    - KISS: additive injection, not attention manipulation
    - Can be disabled via flag for A/B testing
    - No changes to existing understanding graph or retrieval logic
"""

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class UnconsciousInjector:
    """Inject experience vectors into Qwen3's hidden states during forward pass.

    This is the UNCONSCIOUS path: experience influences the model's
    activations without appearing in the prompt text. The model "feels"
    the experience rather than "reads" about it.

    Attributes:
        model: The Qwen3 model to inject into.
        enabled: Global flag to enable/disable injection (for A/B testing).
        injection_strength: Scaling factor for the additive vector (default 0.1).
            Higher = stronger influence but more disruption.
        hook_layer_index: Which transformer layer to hook (default 14 of 28).
        last_injection_log: Dict recording what was injected last time,
            for Introspector to use.
    """

    # Qwen3-0.6B specifics
    QWEN3_NUM_LAYERS = 28
    QWEN3_HIDDEN_SIZE = 896
    BGE_EMBEDDING_DIM = 1024

    def __init__(self, model, enabled: bool = True,
                 injection_strength: float = 0.1,
                 hook_layer_index: int = 14):
        """Initialize the injector.

        Args:
            model: A Qwen3 model (AutoModelForCausalLM instance).
            enabled: Whether injection is active (can toggle for A/B testing).
            injection_strength: Scaling factor for additive injection.
                0.0 = no effect, 1.0 = full vector added. Default 0.1 is
                conservative — strong enough to influence, weak enough not
                to break the model's generation.
            hook_layer_index: Which transformer layer to hook (0-indexed).
                Default 14 = middle of 28 layers. Earlier = more influence
                on subsequent layers. Later = more targeted influence.
        """
        self.model = model
        self.enabled = enabled
        self.injection_strength = injection_strength
        self.hook_layer_index = hook_layer_index

        # State
        self._hook_handle = None
        self._experience_vector = None  # projected to hidden_size
        self._raw_experience_vector = None  # original 1024-dim (for logging)
        self._active = False

        # Injection log — filled each time active() is used
        self.last_injection_log = {
            'active': False,
            'nodes': [],           # list of node IDs
            'node_details': [],    # list of dicts with name, concept, score
            'strength': 0.0,
            'layer': hook_layer_index,
        }

        # Build projection: 1024 → 896 (simple, NOT trained)
        self._projection = self._build_projection()

    def _build_projection(self) -> torch.nn.Linear:
        """Build a simple linear projection from embedding dim to hidden size.

        Since bge-m3 (1024) and Qwen3 hidden (896) have different dims,
        we need a projection. This is a simple truncated identity mapping:
        take the first 896 dimensions of the 1024-dim vector, with
        Xavier-uniform initialization for the remaining mapping.

        This is NOT trained — it's a reasonable initialization that
        preserves most information via the identity-like structure.
        If it works, we can fine-tune the projection later.
        """
        projection = torch.nn.Linear(self.BGE_EMBEDDING_DIM, self.QWEN3_HIDDEN_SIZE, bias=False)

        # Initialize as identity-like: first 896 dims pass through,
        # remaining 128 dims get small random weights
        with torch.no_grad():
            weight = torch.zeros(self.QWEN3_HIDDEN_SIZE, self.BGE_EMBEDDING_DIM)
            # Identity block for overlapping dimensions
            weight[:min(self.QWEN3_HIDDEN_SIZE, self.BGE_EMBEDDING_DIM),
                   :min(self.QWEN3_HIDDEN_SIZE, self.BGE_EMBEDDING_DIM)] = torch.eye(
                min(self.QWEN3_HIDDEN_SIZE, self.BGE_EMBEDDING_DIM)
            )
            # Small random for the extra dimensions
            if self.BGE_EMBEDDING_DIM > self.QWEN3_HIDDEN_SIZE:
                torch.nn.init.xavier_uniform_(
                    weight[:, self.QWEN3_HIDDEN_SIZE:].unsqueeze(0)
                )
                weight[:, self.QWEN3_HIDDEN_SIZE:] *= 0.1  # Scale down extra dims

            projection.weight.copy_(weight)

        # Move to same device as model
        try:
            device = next(self.model.parameters()).device
            projection = projection.to(device)
        except (StopIteration, AttributeError):
            pass  # Will be moved later when model is on device

        projection.eval()  # Not training this
        return projection

    def _compute_experience_vector(self, nodes: list) -> Optional[torch.Tensor]:
        """Compute a single experience vector from multiple UnderstandingNodes.

        Averages the condition_embedding of all nodes into one vector.
        Weighted by each node's confidence/accuracy.

        Args:
            nodes: List of (UnderstandingNode, score) tuples from retrieval.

        Returns:
            1024-dim tensor (on CPU), or None if no valid embeddings.
        """
        if not nodes:
            return None

        vectors = []
        weights = []

        for node, score in nodes:
            emb = node.condition_embedding
            if emb is None:
                continue
            if not emb:  # empty list
                continue

            vec = torch.tensor(emb, dtype=torch.float32)
            vectors.append(vec)

            # Weight: combination of retrieval score and node accuracy
            w = score * node.accuracy
            weights.append(max(w, 0.01))  # floor to avoid zero weight

        if not vectors:
            return None

        # Weighted average
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum()

        stacked = torch.stack(vectors)  # (num_nodes, 1024)
        experience_vector = (stacked * weights_tensor.unsqueeze(1)).sum(dim=0)

        # Normalize to unit length (direction matters more than magnitude)
        norm = experience_vector.norm()
        if norm > 1e-8:
            experience_vector = experience_vector / norm

        return experience_vector

    def _hook_fn(self, module, input, output):
        """Forward hook that adds experience vector to hidden state.

        This is called during model.generate() on each forward pass
        through the hooked layer. It adds the experience vector to
        the hidden state at the LAST token position only.

        Why last position? Because that's the position being predicted.
        Injecting there steers the next-token prediction without
        disrupting the model's understanding of the already-generated
        context.

        Args:
            module: The transformer layer module.
            input: Layer input (unused).
            output: Layer output — tuple of (hidden_states, ...) for
                transformer layers. We modify hidden_states in-place.
        """
        if self._experience_vector is None or not self._active:
            return output

        try:
            # Transformer layer output is typically (hidden_states, attention_weights, ...)
            # hidden_states shape: (batch_size, seq_len, hidden_size)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                return output

            # Clone to avoid in-place modification issues with autograd
            modified = hidden_states.clone()

            # Inject only at the last token position
            # This is the position being predicted — steering here
            # influences what comes next without disrupting context
            modified[:, -1, :] += (
                self._experience_vector * self.injection_strength
            )

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        except Exception as e:
            logger.warning("Injection hook error (falling back to unmodified): %s", e)
            return output

    def active(self, nodes: list):
        """Context manager: activate injection for the duration of a generate() call.

        Usage:
            injector = UnconsciousInjector(qwen3_model)
            with injector.active(experience_nodes):
                output = qwen3_model.generate(input_ids, ...)

        Args:
            nodes: List of (UnderstandingNode, score) tuples from retrieval.
                These are the relevant experiences to inject.

        Returns:
            self (context manager protocol).
        """
        self._active = True
        self._pending_nodes = nodes
        return self

    def __enter__(self):
        """Enter context: compute experience vector and register hook."""
        if not self.enabled:
            logger.debug("UnconsciousInjector disabled — skipping injection")
            self.last_injection_log['active'] = False
            return self

        nodes = getattr(self, '_pending_nodes', [])
        if not nodes:
            logger.debug("No experience nodes to inject")
            self.last_injection_log['active'] = False
            return self

        # Compute raw experience vector (1024-dim)
        raw_vector = self._compute_experience_vector(nodes)
        if raw_vector is None:
            logger.debug("No valid embeddings in experience nodes — skipping injection")
            self.last_injection_log['active'] = False
            return self
        self._raw_experience_vector = raw_vector

        # Project to hidden size (1024 → 896)
        with torch.no_grad():
            projected = self._projection(raw_vector.unsqueeze(0)).squeeze(0)

        # Move to model's device
        try:
            device = next(self.model.parameters()).device
            projected = projected.to(device)
        except (StopIteration, AttributeError):
            pass

        self._experience_vector = projected

        # Register hook on the target layer
        try:
            layers = self.model.model.layers
            if self.hook_layer_index >= len(layers):
                self.hook_layer_index = len(layers) // 2
                logger.warning(
                    "Hook layer index out of range, using middle layer %d",
                    self.hook_layer_index
                )

            target_layer = layers[self.hook_layer_index]
            self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
            self._active = True

            # Update injection log
            self.last_injection_log = {
                'active': True,
                'nodes': [n.id for n, s in nodes],
                'node_details': [
                    {
                        'id': n.id,
                        'name': n.name,
                        'concept': n.concept,
                        'score': float(s),
                        'accuracy': float(n.accuracy),
                        'source': n.source,
                    }
                    for n, s in nodes
                ],
                'strength': self.injection_strength,
                'layer': self.hook_layer_index,
                'vector_norm': float(raw_vector.norm()),
            }

            logger.info(
                "UnconsciousInjector ACTIVE: %d nodes, strength=%.3f, layer=%d",
                len(nodes), self.injection_strength, self.hook_layer_index
            )

        except (AttributeError, IndexError) as e:
            logger.warning("Failed to register injection hook: %s", e)
            self.last_injection_log['active'] = False
            self._active = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: remove hook and clean up."""
        self._active = False

        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        self._experience_vector = None
        self._raw_experience_vector = None
        self._pending_nodes = []

        return False  # Don't suppress exceptions

    def is_active(self) -> bool:
        """Check if injection is currently active."""
        return self._active

    def get_injection_log(self) -> dict:
        """Get the log of the last injection (for Introspector)."""
        return self.last_injection_log.copy()

    def set_enabled(self, enabled: bool):
        """Enable or disable injection (for A/B testing conscious vs unconscious)."""
        self.enabled = enabled
        logger.info("UnconsciousInjector %s", "ENABLED" if enabled else "DISABLED")
