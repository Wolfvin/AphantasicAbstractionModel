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
    - hidden_size = 1024 (read from model.config.hidden_size)
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

# v35: Governance filtering
from governance.states import LifecycleState, EpistemicState

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

        # Read from model config — don't hardcode (Qwen3-0.6B hidden_size=1024)
        self.QWEN3_HIDDEN_SIZE = model.config.hidden_size
        self.QWEN3_NUM_LAYERS = model.config.num_hidden_layers

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
                # Fix: xavier_uniform_ expects a 2D tensor for correct
                # fan_in/fan_out calculation. unsqueeze(0) made it 3D,
                # causing wrong initialization scale.
                torch.nn.init.xavier_uniform_(
                    weight[:, self.QWEN3_HIDDEN_SIZE:]
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
        Weighted by each node's confidence/accuracy AND governance state.

        v35: Nodes that are DEPRECATED or CONTRADICTED are filtered out.
        Only CANDIDATE and STABLE nodes with non-CONTRADICTED epistemic
        state are included in the injection. This ensures that:
          - Deactivated experiences don't influence behavior
          - Contradicted knowledge isn't injected unconsciously
          - Only verified knowledge steers the model's output

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
            # v35: Governance filter — skip non-injectable nodes
            if not self._is_injectable(node):
                logger.debug(
                    "Skipping non-injectable node %s (lifecycle=%s, epistemic=%s)",
                    node.id,
                    getattr(node, 'lifecycle', '?'),
                    getattr(node, 'epistemic', '?'),
                )
                continue

            emb = node.condition_embedding
            if emb is None:
                continue
            if not emb:  # empty list
                continue

            vec = torch.tensor(emb, dtype=torch.float32)
            vectors.append(vec)

            # Weight: combination of retrieval score, accuracy, and governance quality
            # v35: Include seed_scores.overall() in the weight
            base_weight = score * node.accuracy
            governance_bonus = 1.0
            if hasattr(node, 'seed_scores') and node.seed_scores:
                governance_bonus = 0.5 + (node.seed_scores.overall() * 0.5)
            w = base_weight * governance_bonus
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
            output: Layer output. Qwen3DecoderLayer returns a bare
                torch.Tensor (hidden_states). Some older HF models return
                a tuple (hidden_states, present_key_value, ...). We handle
                both cases.
        """
        if self._experience_vector is None or not self._active:
            return output

        try:
            # Qwen3 (transformers ≥5.x): Qwen3DecoderLayer.forward()
            # returns a bare torch.Tensor, NOT a tuple. The old code only
            # handled tuples, making the hook a dead path on Qwen3.
            #
            # Fix: support both tuple and bare-tensor outputs.
            if isinstance(output, tuple):
                hidden_states = output[0]
            elif isinstance(output, torch.Tensor):
                hidden_states = output
            else:
                return output

            # hidden_states shape: (batch_size, seq_len, hidden_size)
            # Clone to avoid in-place modification issues with autograd
            modified = hidden_states.clone()

            # Inject only at the last token position
            # This is the position being predicted — steering here
            # influences what comes next without disrupting context
            #
            # Cast experience vector to match hidden_states dtype.
            # When Qwen3 is loaded with torch.float16, hidden_states are
            # float16 but _experience_vector is float32. Direct += on a
            # float16 tensor with a float32 source raises:
            #   RuntimeError: result type Float can't be cast to Half
            injection = self._experience_vector.to(modified.dtype) * self.injection_strength
            modified[:, -1, :] += injection

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
                        # v35: Governance info
                        'lifecycle': getattr(n, 'lifecycle', LifecycleState.STABLE).value
                            if hasattr(n, 'lifecycle') and n.lifecycle is not None
                            else 'stable',
                        'epistemic': getattr(n, 'epistemic', EpistemicState.OBSERVED).value
                            if hasattr(n, 'epistemic') and n.epistemic is not None
                            else 'observed',
                        'members': [
                            {'role': m.role, 'description': m.description}
                            for m in n.members
                        ] if hasattr(n, 'members') and n.members else [],
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

    def _is_injectable(self, node) -> bool:
        """Check if a node is eligible for unconscious injection.

        v35: Governance filter — only inject nodes that are:
          - Lifecycle: CANDIDATE or STABLE (not NEW, not DEPRECATED)
          - Epistemic: NOT CONTRADICTED
          - Confidence: above minimum (0.2)

        NEW nodes are not injected because they haven't been verified.
        DEPRECATED nodes are not injected because they've been silenced
        ("deactivate, don't delete").
        CONTRADICTED nodes are not injected because they're flagged as wrong.

        This method is graceful — if a node doesn't have governance
        fields (legacy nodes), it defaults to injectable for backward
        compatibility.
        """
        lifecycle = getattr(node, 'lifecycle', None)
        epistemic = getattr(node, 'epistemic', None)
        confidence = getattr(node, 'confidence', 0.5)

        # Backward compatible: nodes without governance fields are injectable
        if lifecycle is None and epistemic is None:
            return confidence >= 0.2

        # Lifecycle check: only CANDIDATE and STABLE
        if lifecycle not in (LifecycleState.CANDIDATE, LifecycleState.STABLE, None):
            return False

        # Epistemic check: not CONTRADICTED
        if epistemic == EpistemicState.CONTRADICTED:
            return False

        # Confidence check
        if confidence < 0.2:
            return False

        return True
