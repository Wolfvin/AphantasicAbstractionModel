"""
Embedding Provider — Pluggable vector embeddings for semantic similarity.

Provides an EmbeddingProvider protocol and several implementations:
1. SentenceTransformerProvider — uses sentence-transformers library (if installed)
2. OpenAIProvider — uses OpenAI embeddings API (if installed)
3. FallbackEmbeddingProvider — simple keyword-based hashing (no external deps)

The bridge can be configured with any provider via set_embedding_provider().

Analogi: Jin Soun bisa mengenali seseorang dari cara berjalan, suara,
atau wajah — tergantung indera mana yang tersedia. Embedding provider
= indera yang bisa "merasakan" kemiripan makna antar konsep.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default embedding dimension for fallback provider
_FALLBACK_DIM = 128


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    An embedding provider converts text into a fixed-length vector
    that captures semantic meaning. Similar texts should produce
    similar vectors (high cosine similarity).

    Analogi: Ini adalah "indera" yang bisa merasakan kemiripan
    makna — bukan kemiripan teks, tapi kemiripan MAKNA.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert text to an embedding vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts to embedding vectors.

        Default implementation calls embed() for each text.
        Subclasses should override for efficiency.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embedding vectors."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this provider."""
        ...


# ---------------------------------------------------------------------------
# FallbackEmbeddingProvider — no external deps
# ---------------------------------------------------------------------------

class FallbackEmbeddingProvider(EmbeddingProvider):
    """Simple keyword-based embedding provider with no external dependencies.

    Uses a hashing approach: each word in the text is hashed to a set of
    positions in the embedding vector, and the values at those positions
    are incremented. This creates a sparse "bag of words" embedding that
    can capture some semantic similarity for common words.

    Not as good as real embeddings, but always available.

    Analogi: Jin Soun tanpa penglihatan — dia masih bisa merasakan
    kemiripan dari "bentuk" kata-kata, meski tidak seakurat melihat
    langsung.
    """

    def __init__(self, dim: int = _FALLBACK_DIM) -> None:
        """Initialize the fallback provider.

        Args:
            dim: Dimension of embedding vectors (default 128).
        """
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """Convert text to a hash-based embedding vector.

        Each word in the text is hashed to 4 positions in the vector,
        and the values at those positions are incremented. The result
        is then L2-normalized.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        vec = [0.0] * self._dim
        words = self._tokenize(text)

        if not words:
            return vec

        for word in words:
            # Hash each word to 4 positions using MD5
            h = hashlib.md5(word.encode("utf-8")).digest()
            for i in range(4):
                idx = struct.unpack("<I", h[i * 4 : (i + 1) * 4])[0] % self._dim
                # Use sign from hash to allow negative contributions
                sign = 1.0 if h[i] % 2 == 0 else -1.0
                vec[idx] += sign * 0.1

        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        """Return the dimension of embedding vectors."""
        return self._dim

    @property
    def name(self) -> str:
        """Return the name of this provider."""
        return "fallback_hash"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        import re
        # Remove common stop words for better signal
        stop_words = {
            "the", "and", "but", "for", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "are", "has",
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
            "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
        }
        words = re.findall(r"[a-zA-Z\u00C0-\u024F\u0400-\u04FF\u4e00-\u9fff]+", text.lower())
        return [w for w in words if len(w) > 2 and w not in stop_words]


# ---------------------------------------------------------------------------
# SentenceTransformerProvider — uses sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using the sentence-transformers library.

    Requires the `sentence-transformers` package to be installed.
    Uses a default model (all-MiniLM-L6-v2) which is fast and effective.

    Analogi: Jin Soun dengan penglihatan sempurna — dia bisa melihat
    kemiripan makna dengan presisi tinggi, bukan hanya tebakan.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with a sentence-transformers model.

        Args:
            model_name: The model to use (default: all-MiniLM-L6-v2).

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("SentenceTransformerProvider initialized with model '%s'", model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single text using sentence-transformers.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Return the dimension of embedding vectors."""
        return self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        """Return the name of this provider."""
        return f"sentence_transformer_{self._model_name}"


# ---------------------------------------------------------------------------
# OpenAIProvider — uses OpenAI embeddings API
# ---------------------------------------------------------------------------

class OpenAIProvider(EmbeddingProvider):
    """Embedding provider using the OpenAI API.

    Requires the `openai` package and a valid API key.

    Analogi: Jin Soun meminjam mata orang lain — kemampuan
    penglihatan yang sangat baik, tapi membutuhkan sumber daya eksternal.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize with OpenAI API credentials.

        Args:
            model: The embedding model to use.
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.

        Raises:
            ImportError: If openai package is not installed.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIProvider. "
                "Install with: pip install openai"
            ) from exc

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        logger.info("OpenAIProvider initialized with model '%s'", model)

    def embed(self, text: str) -> list[float]:
        """Embed a single text using OpenAI API.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [d.embedding for d in response.data]

    @property
    def dimension(self) -> int:
        """Return the dimension of embedding vectors."""
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dims.get(self._model, 1536)

    @property
    def name(self) -> str:
        """Return the name of this provider."""
        return f"openai_{self._model}"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_embedding_provider(provider_name: str = "auto", **kwargs: Any) -> EmbeddingProvider:
    """Factory function to get an embedding provider.

    Tries providers in order of quality:
    1. "sentence_transformer" — best quality, requires sentence-transformers
    2. "openai" — good quality, requires openai + API key
    3. "fallback" — always available, low quality

    With "auto", tries sentence-transformers first, then OpenAI, then fallback.

    Args:
        provider_name: Provider to use ("auto", "sentence_transformer",
            "openai", "fallback").
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        An EmbeddingProvider instance.
    """
    if provider_name == "fallback":
        return FallbackEmbeddingProvider(**kwargs)

    if provider_name == "sentence_transformer":
        try:
            return SentenceTransformerProvider(**kwargs)
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back")
            return FallbackEmbeddingProvider()

    if provider_name == "openai":
        try:
            return OpenAIProvider(**kwargs)
        except ImportError:
            logger.warning("openai not installed, falling back")
            return FallbackEmbeddingProvider()

    # "auto" — try best available
    if provider_name == "auto":
        try:
            return SentenceTransformerProvider(**kwargs)
        except ImportError:
            pass

        try:
            return OpenAIProvider(**kwargs)
        except (ImportError, Exception):
            pass

        logger.info("No external embedding provider available, using fallback")
        return FallbackEmbeddingProvider()

    raise ValueError(f"Unknown embedding provider: {provider_name}")
