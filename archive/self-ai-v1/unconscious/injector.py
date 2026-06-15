# @WHO:   self-ai/src/unconscious/injector.py
# @WHAT:  Inject experience vectors into Qwen3 hidden states during forward pass
# @PART:  self-ai/unconscious
# @ENTRY: UnconsciousInjector

"""UnconsciousInjector — inject experience vectors into Qwen3's hidden states.

Purpose:
    SELF's understanding graph stores experience as UnderstandingNodes with
    bge-m3 condition_embedding vectors (1024-dim). Currently, these only
    influence answers via the conscious path (text prompt injection).

    UnconsciousInjector adds an UNCONSCIOUS path: it projects the embedding
    vectors of relevant UnderstandingNodes into Qwen3's hidden state space,
    combines them via weighted average, and injects the combined vector into
    Qwen3's hidden state during forward pass via a forward hook on a middle
    transformer layer.

    This is activation steering — the model "feels" the experience without
    the experience appearing in the prompt text. The injection is additive
    (not replace), applied only at the last token position (the prediction
    position), and removed after generation completes.

Architecture:
    UnderstandingNode.condition_embedding (1024-dim)
        ↓ per-node projection via self._projection
    projected_vectors (QWEN3_HIDDEN_SIZE-dim each)
        ↓ weighted average by node.confidence
    combined_vector (QWEN3_HIDDEN_SIZE-dim float tensor)
        ↓ register_forward_hook on Qwen3 middle layer
    hidden_state[:, -1, :] += combined_vector
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
    - A linear projection maps embedding dim → hidden_size
    - Projection is initialized as identity; can be trained via ProjectionTrainer
    - If projection_weights.pt exists, trained weights are auto-loaded

Constraint:
    - KISS: additive injection, not attention manipulation
    - Can be disabled via flag for A/B testing
    - No changes to existing understanding graph or retrieval logic
"""

import logging
from datetime import datetime, timezone
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

    v40: Multi-experience injection — each node's embedding is projected
    individually to Qwen3's hidden state space, then combined via weighted
    average (weights = node.confidence). This replaces the previous approach
    of combining first in embedding space and then projecting the combined
    vector. Projecting first allows the projection to handle each experience
    vector's specific characteristics before blending.

    Attributes:
        model: The Qwen3 model to inject into.
        enabled: Global flag to enable/disable injection (for A/B testing).
        injection_strength: Scaling factor for the additive vector (default 0.3).
            Higher = stronger influence but more disruption.
        hook_layer_index: Which transformer layer to hook (default 14 of 28).
        last_injection_log: Dict recording what was injected last time,
            for Introspector to use.
    """

    BGE_EMBEDDING_DIM = 1024

    def __init__(self, model, enabled: bool = True,
                 injection_strength: float = 0.3,
                 hook_layer_index: int = 14):
        """Initialize the injector.

        Args:
            model: A Qwen3 model (AutoModelForCausalLM instance).
            enabled: Whether injection is active (can toggle for A/B testing).
            injection_strength: Scaling factor for additive injection.
                0.0 = no effect, 1.0 = full vector added. Default 0.3 is
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

        # Build projection: embedding_dim → hidden_size
        # Initialized as identity; auto-loads trained weights if available
        self._projection = self._build_projection()
        self._try_load_trained_projection()

        # v40: Multi-experience injection — lazy bge-m3 embedding model
        # Used by _get_node_embedding() when a node lacks condition_embedding.
        # Loaded lazily via model_registry.get_shared_embedding_model().
        self._embedding_model = None
        self._embedding_model_loaded = False

    def _build_projection(self) -> torch.nn.Linear:
        """Build a linear projection from embedding dim to hidden size.

        For Qwen3-0.6B (hidden_size=1024) and bge-m3 (1024-dim),
        this is a square Linear(1024, 1024) initialized as identity.
        For models where dims differ, the overlapping block is identity
        and extra dimensions get small random weights.

        If projection_weights.pt exists, trained weights will be loaded
        after construction (see _try_load_trained_projection).
        """
        projection = torch.nn.Linear(self.BGE_EMBEDDING_DIM, self.QWEN3_HIDDEN_SIZE, bias=False)

        # Initialize as identity-like: overlapping dims pass through,
        # extra dims get small random weights
        with torch.no_grad():
            weight = torch.zeros(self.QWEN3_HIDDEN_SIZE, self.BGE_EMBEDDING_DIM)
            overlap = min(self.QWEN3_HIDDEN_SIZE, self.BGE_EMBEDDING_DIM)
            weight[:overlap, :overlap] = torch.eye(overlap)

            # Small random for the extra dimensions (when embedding > hidden)
            if self.BGE_EMBEDDING_DIM > self.QWEN3_HIDDEN_SIZE:
                torch.nn.init.xavier_uniform_(weight[:, self.QWEN3_HIDDEN_SIZE:])
                weight[:, self.QWEN3_HIDDEN_SIZE:] *= 0.1

            projection.weight.copy_(weight)

        # Move to same device as model
        try:
            device = next(self.model.parameters()).device
            projection = projection.to(device)
        except (StopIteration, AttributeError):
            pass  # Will be moved later when model is on device

        projection.eval()  # Not training this
        return projection

    def _try_load_trained_projection(self):
        """Auto-load trained projection weights if projection_weights.pt exists.

        This is called during __init__ after building the default projection.
        If a trained weights file exists (created by ProjectionTrainer),
        it replaces the identity initialization with trained weights.

        The load is graceful — if the file doesn't exist or dimensions
        mismatch, the default identity projection is kept and a debug
        message is logged. No crash.
        """
        import os

        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'projection_weights.pt'
        )

        if not os.path.exists(weights_path):
            logger.debug(
                "No trained projection weights at %s — using identity init",
                weights_path
            )
            return

        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
            saved_weight = checkpoint['weight']
            saved_config = checkpoint.get('config', {})

            # Validate dimensions
            expected_shape = (self.QWEN3_HIDDEN_SIZE, self.BGE_EMBEDDING_DIM)
            if saved_weight.shape != expected_shape:
                logger.warning(
                    "Projection weights shape mismatch: saved=%s, expected=%s "
                    "— keeping identity init",
                    tuple(saved_weight.shape), expected_shape
                )
                return

            # Load weights
            with torch.no_grad():
                self._projection.weight.data.copy_(saved_weight)

            # Move to model's device
            try:
                device = next(self.model.parameters()).device
                self._projection = self._projection.to(device)
            except (StopIteration, AttributeError):
                pass

            logger.info(
                "Loaded trained projection weights from %s "
                "(shape=%s, trained_layer=%s)",
                weights_path,
                tuple(saved_weight.shape),
                saved_config.get('hook_layer_index', '?'),
            )

        except Exception as e:
            logger.warning(
                "Failed to load projection weights from %s: %s "
                "— keeping identity init",
                weights_path, e
            )

    # ═══════════════ v40: MULTI-EXPERIENCE INJECTION ═══════════════

    def _ensure_embedding_model(self):
        """Lazy-load bge-m3 embedding model from shared singleton.

        Only loaded when needed — i.e., when a node lacks a cached
        condition_embedding and we need to encode its text on-the-fly.
        Uses get_shared_embedding_model() to avoid loading bge-m3
        multiple times (~2.2GB RAM each).
        """
        if self._embedding_model_loaded:
            return

        self._embedding_model_loaded = True  # Mark as attempted

        try:
            from derivation.model_registry import get_shared_embedding_model
            self._embedding_model = get_shared_embedding_model()
            if self._embedding_model is not None:
                logger.debug("Embedding model loaded for multi-experience injection")
            else:
                logger.warning(
                    "bge-m3 not available — nodes without cached embeddings "
                    "will be skipped in multi-experience injection"
                )
        except ImportError:
            logger.warning(
                "model_registry not available — cannot load embedding model"
            )
        except Exception as e:
            logger.warning("Failed to load embedding model: %s", e)

    def _get_node_embedding(self, node) -> Optional[torch.Tensor]:
        """Get the embedding vector for a node, using cache or encoding on-the-fly.

        Checks if the node has a cached condition_embedding (stored as a
        list or tensor). If so, returns it as a float32 tensor. If not,
        encodes the node's concept text via bge-m3 (lazy-loaded).

        This ensures that nodes with pre-computed embeddings don't require
        the embedding model to be loaded, saving ~2.2GB RAM when all nodes
        have cached embeddings.

        Args:
            node: UnderstandingNode instance.

        Returns:
            Float32 tensor of shape (BGE_EMBEDDING_DIM,), or None if
            no embedding is available (neither cached nor encodable).
        """
        # Check for cached embedding — most common path
        emb = getattr(node, 'condition_embedding', None)
        if emb is not None:
            if isinstance(emb, (list, tuple)) and len(emb) > 0:
                return torch.tensor(emb, dtype=torch.float32)
            if isinstance(emb, torch.Tensor) and emb.numel() > 0:
                return emb.float().flatten()

        # No cached embedding — try encoding via bge-m3
        self._ensure_embedding_model()
        if self._embedding_model is None:
            logger.debug(
                "No embedding model — cannot encode node %s",
                getattr(node, 'id', '?')
            )
            return None

        # Use node.concept as the text to encode (most representative)
        text = getattr(node, 'concept', '') or getattr(node, 'name', '') or ''
        if not text:
            logger.debug("Node %s has no text to encode", getattr(node, 'id', '?'))
            return None

        try:
            embedding = self._embedding_model.encode(text, normalize_embeddings=True)
            return torch.tensor(embedding, dtype=torch.float32)
        except Exception as e:
            logger.warning(
                "Failed to encode node %s: %s", getattr(node, 'id', '?'), e
            )
            return None

    def _compute_combined_vector(self, nodes: list) -> Optional[torch.Tensor]:
        """Compute a combined experience vector from multiple UnderstandingNodes.

        v40: Multi-experience injection — projects each node's embedding
        individually to Qwen3's hidden state space and then combines them
        via weighted average. This replaces the previous approach of combining
        first in embedding space and then projecting the combined vector.

        Why project-then-combine instead of combine-then-project?
          - Each experience vector may have different characteristics that the
            projection should handle individually before blending.
          - The projection is trained to map individual embeddings to meaningful
            directions in hidden state space; combining first may produce a
            vector that doesn't correspond to any meaningful direction.
          - Projecting first preserves the directional information of each
            experience, leading to more meaningful activation steering.

        Algorithm:
          1. For each injectable node:
             a. Get experience embedding (cached condition_embedding or
                encode via bge-m3)
             b. Project to hidden state space via self._projection
             c. Record node.confidence as weight
          2. If all weights are 0, use equal weights (fallback)
          3. Compute weighted average of projected vectors
          4. Normalize to unit length
          5. Return single combined tensor (QWEN3_HIDDEN_SIZE-dim)

        Args:
            nodes: List of (UnderstandingNode, score) tuples from retrieval.

        Returns:
            Float32 tensor of shape (QWEN3_HIDDEN_SIZE,) — the combined
            projected experience vector, or None if no valid vectors.
        """
        if not nodes:
            return None

        projected_vectors = []
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

            # Get experience vector — use cached or encode on-the-fly
            vec = self._get_node_embedding(node)
            if vec is None:
                continue

            # Project to hidden state space individually
            with torch.no_grad():
                projected = self._projection(vec.unsqueeze(0)).squeeze(0)

            projected_vectors.append(projected)

            # Weight = node.confidence (not retrieval score)
            confidence = getattr(node, 'confidence', 0.5)
            weights.append(confidence)

        if not projected_vectors:
            return None

        # If all confidences are 0, fallback to equal weights
        total_weight = sum(weights)
        if total_weight == 0:
            logger.debug(
                "All node confidences are 0 — using equal weights for %d nodes",
                len(projected_vectors),
            )
            weights = [1.0] * len(projected_vectors)

        # Weighted average
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum()

        stacked = torch.stack(projected_vectors)  # (num_nodes, QWEN3_HIDDEN_SIZE)
        combined = (stacked * weights_tensor.unsqueeze(1)).sum(dim=0)

        # Normalize to unit length (direction matters more than magnitude)
        norm = combined.norm()
        if norm > 1e-8:
            combined = combined / norm

        return combined

    def _compute_experience_vector(self, nodes: list) -> Optional[torch.Tensor]:
        """Compute a single experience vector from multiple UnderstandingNodes.

        Legacy method: averages condition_embedding of all nodes in embedding
        space, weighted by retrieval score * accuracy * governance bonus.

        This method is preserved for backward compatibility but is no longer
        used in the main injection flow (v40 uses _compute_combined_vector
        instead, which projects each vector individually before combining).

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

        v40: The _experience_vector is now computed via
        _compute_combined_vector(), which projects each node individually
        and then combines via weighted average. The hook itself is
        unchanged — it simply adds the combined vector to the hidden state.

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
        """Enter context: compute combined experience vector and register hook.

        v40: Uses _compute_combined_vector() which projects each node's
        embedding individually and then combines via weighted average
        (weights = node.confidence). This replaces the previous approach
        of combining in embedding space first and then projecting.
        """
        if not self.enabled:
            logger.debug("UnconsciousInjector disabled — skipping injection")
            self.last_injection_log['active'] = False
            return self

        nodes = getattr(self, '_pending_nodes', [])
        if not nodes:
            logger.debug("No experience nodes to inject")
            self.last_injection_log['active'] = False
            return self

        # v40: Compute combined experience vector via multi-experience injection
        # Projects each node individually, then combines via weighted average
        combined_vector = self._compute_combined_vector(nodes)
        if combined_vector is None:
            logger.debug("No valid embeddings in experience nodes — skipping injection")
            self.last_injection_log['active'] = False
            return self

        self._raw_experience_vector = combined_vector.clone()

        # Move to model's device
        try:
            device = next(self.model.parameters()).device
            combined_vector = combined_vector.to(device)
        except (StopIteration, AttributeError):
            pass

        self._experience_vector = combined_vector

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
                'vector_norm': float(combined_vector.norm()),
            }

            logger.info(
                "UnconsciousInjector ACTIVE: %d nodes, strength=%.3f, layer=%d",
                len(nodes), self.injection_strength, self.hook_layer_index
            )

            # v36: Mark all injectable nodes as used (for temporal decay tracking)
            now_iso = datetime.now(timezone.utc).isoformat()
            for n, s in nodes:
                if self._is_injectable(n) and hasattr(n, 'last_used_at'):
                    n.last_used_at = now_iso

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
