# @WHO:   self-ai/src/derivation/embedding_concepts.py
# @WHAT:  Embedding-based concept detection using BAAI/bge-m3
# @PART:  self-ai/derivation
# @ENTRY: EmbeddingConceptDetector

"""Embedding-Based Concept Detection — replaces CONCEPT_CLUSTERS keyword lookup.

Uses BAAI/bge-m3 sentence embeddings to compute concept cluster centroids,
then detects concepts via cosine similarity between text embedding and centroids.

This follows the "NO hardcoded rules" philosophy: concepts are detected by
semantic similarity, not by regex word-boundary matching.

Centroids are precomputed and cached to disk so subsequent runs are fast.
If torch/sentence-transformers is unavailable, the caller should fall back
to keyword matching (graceful degradation).
"""

import os
import json
import logging
import re

import numpy as np

# CONCEPT_CLUSTERS was removed — SELF will build clusters through teaching.
# Use empty dict as default; embedding centroids will be computed when
# concept_clusters are provided at runtime.
try:
    from derivation.concepts import CONCEPT_CLUSTERS
except ImportError:
    CONCEPT_CLUSTERS = {}  # SELF will build clusters through teaching

logger = logging.getLogger(__name__)


class EmbeddingConceptDetector:
    """Detect concepts in text using embedding similarity against precomputed centroids.

    API mirrors the keyword-based methods:
      - detect_concepts(text) -> dict  {concept_path: [matched_words]}
      - has_concept(text, concept_path) -> bool

    Usage:
        detector = EmbeddingConceptDetector()
        concepts = detector.detect_concepts("Hujan turun deras, petir menyambar")
        # -> {'weather_sign.water': ['hujan'], 'weather_sign.thunder': ['petir'], ...}

        if detector.has_concept("Angin bertiup kencang", "weather_sign.wind"):
            ...
    """

    # Model identifier for bge-m3
    MODEL_NAME = 'BAAI/bge-m3'

    def __init__(self, concept_clusters=None, cache_dir=None):
        """Initialize the detector.

        Args:
            concept_clusters: Dict of {cluster_name: {subcluster_name: [words]}}.
                              Defaults to CONCEPT_CLUSTERS from concepts.py.
            cache_dir: Directory for caching centroids. Defaults to
                       <project_root>/data/embedding_cache/
        """
        self.model = None  # Lazy loaded — only when needed
        self.centroids = {}  # concept_path -> np.array (1024-dim, normalized)
        self.cluster_names = {}  # concept_path -> list of words
        self._loaded = False

        # Determine cache directory
        if cache_dir is not None:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'embedding_cache'
            )

        self.concept_clusters = concept_clusters or CONCEPT_CLUSTERS

    # ── Public API ──────────────────────────────────────────────

    def detect_concepts(self, text: str, top_k: int = 5, threshold: float = 0.3) -> dict:
        """Detect concepts in text using embedding similarity.

        Returns dict of {concept_path: [matched_words]} — same format as
        the keyword-based _detect_concepts method in TextComprehension.
        """
        self._ensure_loaded()
        self._ensure_model()  # Need the model for encoding new text

        if self.model is None:
            # Could not load model — return empty (caller should fall back)
            return {}

        # Encode the full text
        text_emb = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]

        # Compare against all centroids
        scores = {}
        for path, centroid in self.centroids.items():
            sim = float(np.dot(text_emb, centroid))
            if sim >= threshold:
                scores[path] = sim

        # Sort by similarity and take top_k
        sorted_paths = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Format result same as keyword-based _detect_concepts
        result = {}
        text_lower = text.lower()
        for path, score in sorted_paths:
            words = self.cluster_names.get(path, [])
            found = []
            for w in words:
                if len(w) >= 3:
                    if re.search(r'\b' + re.escape(w) + r'\b', text_lower):
                        found.append(w)
                elif w in text_lower:
                    found.append(w)

            # If no exact keyword match but embedding says it's similar,
            # add the top words as approximate match (semantic detection)
            if not found and score > 0.4:
                found = words[:3]  # Top 3 words as approximate match

            if found:
                result[path] = found

        return result

    def has_concept(self, text: str, concept_path: str, threshold: float = 0.3) -> bool:
        """Check if text contains a specific concept via embedding similarity."""
        self._ensure_loaded()
        self._ensure_model()

        if concept_path not in self.centroids:
            return False

        if self.model is None:
            return False

        text_emb = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
        centroid = self.centroids[concept_path]
        sim = float(np.dot(text_emb, centroid))
        return sim >= threshold

    # ── Lazy Loading ────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy load model and centroids — only when first used.

        Uses model_registry.get_shared_embedding_model() to avoid
        loading bge-m3 multiple times (~2.2GB RAM each).
        """
        if self._loaded:
            return

        # Try loading from cache first
        if self._load_centroids_from_cache():
            logger.info("Loaded concept centroids from cache (%d centroids)", len(self.centroids))
            # Still need model for on-demand encoding
            self._ensure_model()
            self._loaded = True
            return

        # Otherwise compute centroids from scratch
        try:
            from derivation.model_registry import get_shared_embedding_model
            self.model = get_shared_embedding_model()
            if self.model is None:
                logger.warning(
                    "Shared embedding model not available — "
                    "embedding concept detection disabled. Falling back to keyword matching."
                )
                self._loaded = True
                return
            logger.info("Computing concept centroids using shared bge-m3 ...")
            self._compute_centroids()
            self._save_centroids_to_cache()
            logger.info("Computed and cached %d concept centroids", len(self.centroids))
        except ImportError:
            logger.warning(
                "model_registry not available — "
                "embedding concept detection disabled. Falling back to keyword matching."
            )
            self._loaded = True  # Don't retry
            return
        except Exception as e:
            logger.warning("Failed to get shared embedding model: %s — falling back to keywords.", e)
            self._loaded = True  # Don't retry
            return

        self._loaded = True

    def _ensure_model(self):
        """Ensure the embedding model is loaded via shared singleton.

        Uses model_registry.get_shared_embedding_model() to avoid
        loading bge-m3 multiple times.
        """
        if self.model is not None:
            return
        try:
            from derivation.model_registry import get_shared_embedding_model
            self.model = get_shared_embedding_model()
            if self.model is not None:
                logger.info("EmbeddingConceptDetector using shared bge-m3")
            else:
                logger.warning("Shared embedding model not available for concept detection")
        except Exception as e:
            logger.warning("Failed to get shared embedding model: %s", e)
            self.model = None

    # ── Centroid Computation ────────────────────────────────────

    def _compute_centroids(self):
        """Compute centroid for each sub-cluster by averaging word embeddings."""
        self._ensure_model()
        if self.model is None:
            return

        for cn, subs in self.concept_clusters.items():
            for sn, words in subs.items():
                # Handle special nested structure (e.g., contextual_meaning_map)
                if isinstance(words, dict):
                    # Nested dict — flatten with extra key level
                    for sn2, inner_words in words.items():
                        path = f"{cn}.{sn}.{sn2}"
                        if isinstance(inner_words, list) and len(inner_words) > 0:
                            self._compute_single_centroid(path, inner_words)
                    continue

                if not isinstance(words, list) or len(words) == 0:
                    continue

                path = f"{cn}.{sn}"
                self._compute_single_centroid(path, words)

    def _compute_single_centroid(self, path: str, words: list):
        """Compute and store a single centroid for a list of words."""
        try:
            embeddings = self.model.encode(words, show_progress_bar=False, normalize_embeddings=True)
            # Average to get centroid
            centroid = np.mean(embeddings, axis=0)
            # Re-normalize
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid = centroid / norm
            self.centroids[path] = centroid
            self.cluster_names[path] = words
        except Exception as e:
            logger.warning("Failed to compute centroid for %s: %s", path, e)

    # ── Cache I/O ──────────────────────────────────────────────

    def _save_centroids_to_cache(self):
        """Save centroids and cluster names to disk cache."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            # Save centroids as compressed npz
            cache_path = os.path.join(self.cache_dir, 'concept_centroids.npz')
            if self.centroids:
                paths = list(self.centroids.keys())
                centroid_stack = np.stack([self.centroids[p] for p in paths])
                np.savez_compressed(cache_path, paths=paths, centroids=centroid_stack)

            # Save cluster_names separately as JSON
            names_path = os.path.join(self.cache_dir, 'concept_cluster_names.json')
            with open(names_path, 'w', encoding='utf-8') as f:
                json.dump(self.cluster_names, f, ensure_ascii=False, indent=2)

            logger.info("Saved centroids to %s", self.cache_dir)
        except Exception as e:
            logger.warning("Failed to save centroid cache: %s", e)

    def _load_centroids_from_cache(self) -> bool:
        """Load centroids from disk cache. Returns True if successful."""
        cache_path = os.path.join(self.cache_dir, 'concept_centroids.npz')
        names_path = os.path.join(self.cache_dir, 'concept_cluster_names.json')

        if not os.path.exists(cache_path) or not os.path.exists(names_path):
            return False

        try:
            data = np.load(cache_path, allow_pickle=False)
            paths = data['paths']
            centroids = data['centroids']

            for path, centroid in zip(paths, centroids):
                self.centroids[str(path)] = centroid

            with open(names_path, 'r', encoding='utf-8') as f:
                self.cluster_names = json.load(f)

            return len(self.centroids) > 0
        except Exception as e:
            logger.warning("Failed to load centroid cache: %s", e)
            return False
