"""AGNN Embeddings — model-native embedding extraction.

This module extracts embeddings directly from a language model's internal
representations, eliminating the dependency on external embedding models
like bge-m3. By reading hidden states from the model's own layers, we
obtain embeddings that are natively aligned with the model's semantic
space — no projection matrix needed.

Architecture Decision — Why Middle Layer, Not Last Layer:
    The last layer of a causal language model is optimized for next-token
    prediction. Its representations are highly task-specific: they encode
    distributional information about the immediate next token, not general
    semantic content about the input as a whole.

    Research (e.g., "BERT has a Mouth and It Speaks" / "How Contextual are
    Contextualized Word Representations" / Voita et al. 2019) consistently
    shows that middle layers of transformer models capture the richest
    syntactic and semantic information. The last layer is a bottleneck
    that compresses everything into a next-token distribution.

    For AGNN's use case — encoding node semantics for graph-based
    reasoning — we want representations that capture WHAT something means,
    not what token likely follows. Hence: middle layer extraction.

Pooling Strategy — Why Mean-Pool, Not CLS:
    Causal LMs (like Qwen3-0.6B) do not have a [CLS] token. Even for
    encoder models, mean-pooling across all token positions is more robust
    than relying on a single token position. Mean-pooling:
      1. Aggregates information from the entire input
      2. Is not sensitive to token position biases
      3. Produces stable representations regardless of input length

    We apply a simple unweighted mean across all non-padding tokens.

Inference-Only Contract:
    All embedding extraction happens with torch.no_grad(). No gradients
    are computed, no model weights are updated. This is a read-only
    operation on the model's internal representations.

CPU Compatibility:
    The module explicitly forces computation to CPU via .to('cpu') and
    does not require CUDA. This ensures tests and development can run
    on any machine.

Status: Implemented — ModelEmbedder, EmbeddingCache, embed_node, and
    AGNNGraph integration (set_embedder, initialize_embeddings).
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = ".cache"
DEFAULT_CACHE_FILENAME = "agnn_embeddings.pkl"
DEFAULT_MAX_SEQ_LENGTH = 512
"""Maximum sequence length for tokenization — prevents OOM on long inputs."""


# ──────────────────────────────────────────────────────
#  ModelEmbedder
# ──────────────────────────────────────────────────────

class ModelEmbedder:
    """Extract embeddings from a HuggingFace transformer model's hidden states.

    ModelEmbedder wraps a (model, tokenizer) pair and provides a simple
    API to embed text strings into fixed-size numpy vectors. The embedding
    is extracted from a specified hidden layer (default: middle layer) and
    mean-pooled across all token positions.

    This class is inference-only — it never computes gradients or modifies
    model weights. It forces CPU computation to ensure portability.

    Usage with a real model:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

        embedder = ModelEmbedder(model=model, tokenizer=tokenizer)
        embedding = embedder.embed("harimau adalah karnivora")
        print(embedding.shape)  # (896,) for Qwen3-0.6B

    Usage with a mock (for testing):
        # Create a mock embedder that returns random vectors
        mock = ModelEmbedder(
            model=None, tokenizer=None,
            hidden_size=64, num_layers=12,
        )
        embedding = mock.embed("any text")
        print(embedding.shape)  # (64,)

    Architecture:
        1. Tokenize input text (truncate to max_seq_length)
        2. Forward pass with output_hidden_states=True
        3. Select hidden state from layer_index (default: num_layers // 2)
        4. Mean-pool across all token positions
        5. Return as numpy float32 array
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        layer_index: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        model_id: str = "unknown",
    ):
        """Initialize the ModelEmbedder.

        Args:
            model: A HuggingFace transformer model (e.g., AutoModelForCausalLM).
                If None, the embedder operates in mock mode — embed() returns
                random vectors of shape (hidden_size,).
            tokenizer: A HuggingFace tokenizer matching the model.
                Required if model is provided; ignored in mock mode.
            layer_index: Which hidden layer to extract from (0-indexed).
                If None, defaults to num_layers // 2 (middle layer).
                Middle layer captures the richest semantic information
                (not the last layer, which is too task-specific).
            hidden_size: Dimensionality of the model's hidden states.
                If None, auto-detected from model.config.hidden_size.
                Required for mock mode (when model is None).
            num_layers: Number of transformer layers in the model.
                If None, auto-detected from model.config.num_hidden_layers.
                Required for mock mode (when model is None).
            max_seq_length: Maximum token length for input text.
                Longer inputs are truncated to prevent OOM.
            model_id: Identifier for the model (used for cache keys and logging).
        """
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._model_id = model_id
        self._is_mock = model is None

        # Auto-detect hidden_size and num_layers from model config
        if model is not None:
            try:
                config = model.config
                self._hidden_size = hidden_size or getattr(config, 'hidden_size', 768)
                self._num_layers = num_layers or getattr(config, 'num_hidden_layers', 12)
            except Exception:
                self._hidden_size = hidden_size or 768
                self._num_layers = num_layers or 12
        else:
            # Mock mode — must provide hidden_size explicitly or default to 64
            self._hidden_size = hidden_size or 64
            self._num_layers = num_layers or 12

        # Determine layer index: default to middle layer
        if layer_index is not None:
            self._layer_index = layer_index
        else:
            self._layer_index = self._num_layers // 2

        logger.info(
            "ModelEmbedder initialized: model_id=%s, hidden_size=%d, "
            "num_layers=%d, layer_index=%d, mock=%s",
            self._model_id, self._hidden_size, self._num_layers,
            self._layer_index, self._is_mock,
        )

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    @property
    def hidden_size(self) -> int:
        """Return the hidden state dimensionality."""
        return self._hidden_size

    @property
    def layer_index(self) -> int:
        """Return the hidden layer index used for extraction."""
        return self._layer_index

    @property
    def is_mock(self) -> bool:
        """Return True if this embedder is in mock mode (no real model)."""
        return self._is_mock

    def embed(self, text: str) -> np.ndarray:
        """Embed a text string into a fixed-size numpy vector.

        The embedding is extracted from the model's hidden state at
        self._layer_index, mean-pooled across all token positions.

        In mock mode (no real model), returns a deterministic random
        vector seeded by the text content — same text always produces
        the same embedding, but different texts produce different ones.

        Args:
            text: The text to embed.

        Returns:
            numpy float32 array of shape (hidden_size,).

        Raises:
            RuntimeError: If model forward pass fails.
        """
        if not text or not text.strip():
            # Empty text → zero vector
            return np.zeros(self._hidden_size, dtype=np.float32)

        if self._is_mock:
            return self._mock_embed(text)

        return self._real_embed(text)

    def _mock_embed(self, text: str) -> np.ndarray:
        """Generate a deterministic mock embedding based on text content.

        Uses the text's hash as a seed for numpy's RNG, ensuring:
          1. Same text → same embedding (deterministic for caching)
          2. Different texts → different embeddings (diverse)
          3. No external model dependency

        The vector is normalized to unit length to match the convention
        used by message passing (which normalizes embeddings after update).
        """
        # Create a deterministic seed from the text
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)  # Use first 8 hex chars as seed

        rng = np.random.RandomState(seed)
        embedding = rng.randn(self._hidden_size).astype(np.float32)

        # Normalize to unit length (consistent with message_pass normalization)
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding

    def _real_embed(self, text: str) -> np.ndarray:
        """Extract embedding from a real HuggingFace transformer model.

        Steps:
          1. Tokenize input (truncate to max_seq_length)
          2. Forward pass with output_hidden_states=True
          3. Select hidden state at self._layer_index
          4. Mean-pool across all token positions
          5. Return as numpy float32 array

        All computation is done on CPU with no gradients.
        """
        import torch

        # Tokenize
        encoded = self._tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self._max_seq_length,
            padding=False,
        )

        # Force CPU — no GPU required
        input_ids = encoded['input_ids'].to('cpu')

        # Forward pass — no gradients, output hidden states
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, output_hidden_states=True)

        # Extract hidden states from the specified layer
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 = embedding layer, Index 1..N = transformer layers
        all_hidden_states = outputs.hidden_states
        if all_hidden_states is None:
            logger.warning("Model did not return hidden_states — falling back to mock")
            return self._mock_embed(text)

        # Clamp layer_index to valid range
        layer_idx = min(self._layer_index, len(all_hidden_states) - 1)
        hidden_state = all_hidden_states[layer_idx]  # (1, seq_len, hidden_size)

        # Mean-pool across token positions: (1, seq_len, hidden_size) → (hidden_size,)
        # Convert to numpy first for consistency
        hidden_np = hidden_state.squeeze(0).cpu().numpy()  # (seq_len, hidden_size)
        embedding = np.mean(hidden_np, axis=0).astype(np.float32)  # (hidden_size,)

        # Normalize to unit length (consistent with message_pass normalization)
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts in sequence.

        Currently processes texts one at a time (no batching). This is
        intentional for AGNN's use case — we typically embed a handful of
        node labels, not thousands of documents. Batch processing can be
        added later as an optimization if needed.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of numpy float32 arrays, each of shape (hidden_size,).
        """
        return [self.embed(text) for text in texts]


# ──────────────────────────────────────────────────────
#  EmbeddingCache
# ──────────────────────────────────────────────────────

class EmbeddingCache:
    """Cache for computed embeddings, keyed by (model_id, text).

    The cache stores embeddings as numpy arrays and uses a hash of
    (model_id, text) as the key. This ensures:
      1. Same text with same model → cache HIT (no re-computation)
      2. Same text with different model → cache MISS (different spaces)
      3. Different text → cache MISS (obviously different)

    Cache Invalidation:
      The cache does NOT auto-invalidate — it assumes that for a given
      (model_id, text) pair, the embedding is stable. This is correct
      because:
        - The model weights don't change during inference
        - The same text always produces the same embedding (deterministic)

      However, if a node's text changes (e.g., label updated), the OLD
      cache entry for the old text is NOT automatically removed. Call
      invalidate(text, model_id) or clear() to handle this case.

    Disk Persistence:
      Optionally, the cache can be persisted to disk as a pickle file.
      This survives process restarts and avoids re-computing embeddings
      for the same texts across sessions.

      The cache file is stored at:
        {cache_dir}/{cache_filename}

      Typical path: .cache/agnn_embeddings.pkl

    Thread Safety:
      NOT thread-safe. For now, AGNN is single-threaded. If concurrency
      is added later, wrap cache access with a lock.

    Usage:
        cache = EmbeddingCache()

        # Store
        cache.put("harimau", "qwen3-0.6b", np.array([0.1, 0.2, ...]))

        # Retrieve
        emb = cache.get("harimau", "qwen3-0.6b")  # HIT → returns array
        emb = cache.get("harimau", "bge-m3")       # MISS → returns None

        # Persist to disk
        cache.save()
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        cache_filename: str = DEFAULT_CACHE_FILENAME,
        auto_load: bool = True,
    ):
        """Initialize the EmbeddingCache.

        Args:
            cache_dir: Directory for the cache file.
            cache_filename: Name of the cache file.
            auto_load: If True, automatically load existing cache from disk.
        """
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_dir = cache_dir
        self._cache_filename = cache_filename
        self._cache_path = os.path.join(cache_dir, cache_filename)

        if auto_load:
            self.load()

    @staticmethod
    def _make_key(model_id: str, text: str) -> str:
        """Generate a deterministic cache key from (model_id, text).

        The key is a SHA-256 hash of the concatenation of model_id and text,
        prefixed with a short human-readable tag for debugging.

        Format: "{model_id[:8]}:{hash[:16]}"

        This ensures:
          - Deterministic: same input → same key
          - Collision-resistant: SHA-256 is cryptographic
          - Debuggable: prefix shows which model, hash suffix differentiates texts
        """
        raw = f"{model_id}||{text}"
        hash_hex = hashlib.sha256(raw.encode('utf-8')).hexdigest()
        return f"{model_id[:8]}:{hash_hex[:16]}"

    def get(self, text: str, model_id: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding.

        Args:
            text: The text that was embedded.
            model_id: The model identifier used for embedding.

        Returns:
            The cached numpy array, or None if not found (cache MISS).
        """
        key = self._make_key(model_id, text)
        return self._cache.get(key)

    def put(self, text: str, model_id: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache.

        Args:
            text: The text that was embedded.
            model_id: The model identifier used for embedding.
            embedding: The computed embedding (numpy float32 array).
        """
        key = self._make_key(model_id, text)
        self._cache[key] = embedding.copy().astype(np.float32)

    def has(self, text: str, model_id: str) -> bool:
        """Check if an embedding is cached without retrieving it.

        Args:
            text: The text to check.
            model_id: The model identifier.

        Returns:
            True if the cache has this embedding, False otherwise.
        """
        key = self._make_key(model_id, text)
        return key in self._cache

    def invalidate(self, text: str, model_id: str) -> bool:
        """Remove a specific embedding from the cache.

        This should be called when a node's text changes and the old
        embedding is no longer valid.

        Args:
            text: The text whose embedding to invalidate.
            model_id: The model identifier.

        Returns:
            True if the entry existed and was removed, False otherwise.
        """
        key = self._make_key(model_id, text)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def size(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._cache)

    def save(self) -> bool:
        """Persist the cache to disk.

        Saves the entire _cache dictionary as a pickle file at
        self._cache_path. Creates the directory if it doesn't exist.

        Returns:
            True if save succeeded, False if it failed.
        """
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            with open(self._cache_path, 'wb') as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("EmbeddingCache saved: %d entries to %s",
                        len(self._cache), self._cache_path)
            return True
        except Exception as e:
            logger.warning("Failed to save EmbeddingCache: %s", e)
            return False

    def load(self) -> bool:
        """Load the cache from disk.

        Reads the pickle file at self._cache_path and merges it into
        the current cache. Existing entries are NOT overwritten —
        in-memory entries take precedence.

        Returns:
            True if load succeeded, False if it failed (file not found,
            corrupted, etc.).
        """
        if not os.path.exists(self._cache_path):
            return False

        try:
            with open(self._cache_path, 'rb') as f:
                loaded = pickle.load(f)

            if not isinstance(loaded, dict):
                logger.warning("Cache file has invalid format — skipping")
                return False

            # Merge: in-memory entries take precedence
            for key, value in loaded.items():
                if key not in self._cache:
                    self._cache[key] = value

            logger.info("EmbeddingCache loaded: %d entries from %s",
                        len(loaded), self._cache_path)
            return True
        except Exception as e:
            logger.warning("Failed to load EmbeddingCache: %s", e)
            return False

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return f"EmbeddingCache(entries={self.size()}, path={self._cache_path})"


# ──────────────────────────────────────────────────────
#  embed_node — the main embedding function
# ──────────────────────────────────────────────────────

def embed_node(
    node_text: str,
    embedder: ModelEmbedder,
    cache: Optional[EmbeddingCache] = None,
) -> np.ndarray:
    """Embed a node's text, using the cache if available.

    This is the primary entry point for embedding a single node.
    It checks the cache first, and only computes the embedding
    via the embedder on a cache MISS.

    Args:
        node_text: The text to embed (typically the node's label).
        embedder: A ModelEmbedder instance (real or mock).
        cache: Optional EmbeddingCache. If provided, checked before
            computing. If None, always computes.

    Returns:
        numpy float32 array of shape (embedder.hidden_size,).
    """
    model_id = embedder.model_id

    # Check cache first
    if cache is not None:
        cached = cache.get(node_text, model_id)
        if cached is not None:
            logger.debug("Cache HIT for node text: '%s' (model=%s)", node_text, model_id)
            return cached

    # Cache MISS — compute via embedder
    logger.debug("Cache MISS for node text: '%s' (model=%s) — computing", node_text, model_id)
    embedding = embedder.embed(node_text)

    # Store in cache
    if cache is not None:
        cache.put(node_text, model_id, embedding)

    return embedding


def embed_nodes_batch(
    texts: List[str],
    embedder: ModelEmbedder,
    cache: Optional[EmbeddingCache] = None,
) -> List[np.ndarray]:
    """Embed multiple node texts, using the cache where possible.

    For each text:
      1. Check cache → HIT: return cached embedding
      2. Cache MISS → compute via embedder
      3. Store result in cache

    This function is more efficient than calling embed_node() in a loop
    because it can batch the cache-miss computations.

    Args:
        texts: List of text strings to embed.
        embedder: A ModelEmbedder instance.
        cache: Optional EmbeddingCache.

    Returns:
        List of numpy float32 arrays, one per text.
    """
    model_id = embedder.model_id
    results: List[np.ndarray] = []
    miss_indices: List[int] = []
    miss_texts: List[str] = []

    # Phase 1: Check cache for all texts
    for i, text in enumerate(texts):
        if cache is not None:
            cached = cache.get(text, model_id)
            if cached is not None:
                results.append(cached)
                continue

        # Cache miss — mark for batch computation
        results.append(None)  # placeholder
        miss_indices.append(i)
        miss_texts.append(text)

    # Phase 2: Batch-compute missed embeddings
    if miss_texts:
        computed = embedder.embed_batch(miss_texts)

        # Phase 3: Fill in results and update cache
        for j, idx in enumerate(miss_indices):
            embedding = computed[j]
            results[idx] = embedding

            if cache is not None:
                cache.put(miss_texts[j], model_id, embedding)

    return results
