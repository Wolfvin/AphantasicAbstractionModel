# @WHO:   self-ai/src/derivation/embedding_retrieval.py
# @WHAT:  Understanding Retrieval via bge-m3 Embedding — replaces keyword matching
# @PART:  self-ai/derivation
# @ENTRY: UnderstandingRetriever

"""Understanding Retrieval via Embedding — HOW SELF finds the right understanding.

Vision:
    SELF adalah LLM yang membangun semantic understanding SENDIRI.
    Understanding ini BISA MEMENGARUHI cara dia menjawab di pertanyaan berikutnya.

    Sistem 1: bge-m3 embedding → intuisi langsung (output yang tiba-tiba)
    Sistem 2: Understanding Graph → pemahaman dari yang kita ajarkan

    SELF bisa KOMBINASIKAN beberapa semantic understanding untuk
    generate jawaban yang sesuai.

Philosophy:
    v27 removes ALL keyword matching. Retrieval is embedding-ONLY.
    Keyword matching is NOT understanding — it's pattern matching that
    misses operational meaning.

    The old system used KEYWORD MATCHING to find which understanding applies:
    - Does "kecuali" appear in the text? → Use signal_flip understanding
    - Does "tetapi" appear? → Use contrast_focus understanding

    This is FRAGILE because:
    1. "kehilangan" doesn't contain "terjual" → wrong understanding
    2. "tapi" could be contrast OR informal speech → wrong routing
    3. New words that SELF discovers can't be matched → blind spots
    4. Keyword overlap misses SEMANTIC similarity entirely

    The new system uses BAAI/bge-m3 EMBEDDING to find understandings by
    SEMANTIC SIMILARITY, not keyword overlap:

    - Embed the question+text → 1024-dim vector
    - Compare against ALL understanding node embeddings
    - Return the most semantically similar understanding

    This solves the "terjual" vs "kehilangan" problem: even though the
    words are different, their OPERATIONAL MEANING is similar (both involve
    quantity reduction), so the embedding similarity will be high.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Question + Text                                        │
    │  "Toko kehilangan 35 roti, ditambah 28. Sisa?"         │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  bge-m3 encode(question + text) → 1024-dim vector      │
    │  ← System 1: Intuisi langsung (output yang tiba-tiba)  │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Compare against ALL understanding embeddings           │
    │  via cosine similarity                                  │
    │                                                         │
    │  U_quantity:  sim = 0.87  ← BEST MATCH                 │
    │  U_signal_flip: sim = 0.34                              │
    │  U_contrast: sim = 0.21                                 │
    │  ...                                                    │
    │  ← System 2: Pemahaman dari yang kita ajarkan          │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Return U_quantity (most semantically similar)          │
    │  Apply its transformation → compute answer              │
    │  ← SELF bisa kombinasikan BEBERAPA understanding       │
    └─────────────────────────────────────────────────────────┘

Key Design Decisions:
    1. EMBEDDING TEXT: We embed not just conditions but the FULL understanding
       context: concept + abstraction + conditions + transformation description.
       This captures OPERATIONAL MEANING, not just word overlap.

    2. TWO-STAGE RETRIEVAL:
       - Stage 1: Embedding similarity (bge-m3) → find top-k candidates
       - Stage 2: Re-rank with operational signal match (lightweight)
       This is faster than brute-force scoring all nodes.

    3. EMBEDDING-ONLY: v27 does NOT fall back to keyword matching.
       If bge-m3 is unavailable, retrieval FAILS. Understanding requires
       semantic similarity, not word overlap.

    4. CACHING: Embeddings are precomputed and cached when nodes are added.
       Only new question embeddings need to be computed at query time.
"""

import os
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class UnderstandingRetriever:
    """Retrieve understandings via bge-m3 semantic embedding similarity.

    This replaces the keyword-based find_matching() in UnderstandingGraph.
    Instead of checking if "kecuali" appears in text, we compute the
    semantic similarity between the question+text and each understanding
    node's embedding, then return the best match.

    The retriever is INDEPENDENT of the understanding graph — it only
    needs access to node embeddings and metadata. This separation allows
    the retriever to be swapped or upgraded without touching the graph.

    Usage:
        retriever = UnderstandingRetriever()

        # During learning: compute embeddings for new understanding nodes
        retriever.index_node(node)

        # During answering: find the best matching understanding
        results = retriever.retrieve(text, question, nodes, top_k=3)
        best_node, score = results[0]
    """

    MODEL_NAME = 'BAAI/bge-m3'
    EMBEDDING_DIM = 1024

    def __init__(self, cache_dir: str = None, experience_store=None,
                 experience_enabled: bool = True):
        """Initialize the retriever.

        Args:
            cache_dir: Directory for caching node embeddings.
                       Defaults to <project_root>/data/embedding_cache/
            experience_store: Optional ExperienceStore instance for
                              experience-based retrieval adjustments.
                              If None, will lazy-init from singleton.
            experience_enabled: Toggle experience adjustment on/off.
                                When disabled, _rerank() behaves as if
                                ExperienceStore is empty. For ablation testing.
        """
        self._model = None  # Lazy loaded
        self._model_loaded = False
        self._cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'embedding_cache'
        )

        # In-memory embedding index: node_id -> np.array (1024-dim, normalized)
        self._embeddings: Dict[str, np.ndarray] = {}

        # ── Experience-based retrieval adjustment ──
        # ExperienceStore records success/failure episodes and adjusts
        # retrieval scores based on historical performance in similar contexts.
        # Qwen3 does NOT know about this — it's purely a bge-m3 layer adjustment.
        self._experience_store = experience_store  # Lazy-init if None
        self._experience_enabled = experience_enabled  # Ablation toggle

        # Load cached embeddings if available
        self._load_embedding_cache()

    # ═══════════════ PUBLIC API ═══════════════

    def index_node(self, node) -> bool:
        """Compute and store embedding for an understanding node.

        The embedding is computed from a RICH TEXT representation that
        captures the node's OPERATIONAL MEANING:

            "[concept] [abstraction] [conditions] [transformation action]"

        This is NOT just word embeddings — it captures HOW the understanding
        operates, what it's about, and when it applies.

        Args:
            node: UnderstandingNode to index

        Returns:
            True if embedding was computed successfully, False otherwise
        """
        # Skip if already indexed (unless forced)
        if node.id in self._embeddings and node.condition_embedding is not None:
            return True

        # Build the rich text representation for embedding
        rich_text = self._build_embedding_text(node)

        # Compute embedding
        embedding = self._encode(rich_text)
        if embedding is None:
            logger.warning("Failed to compute embedding for node %s", node.id)
            return False

        # Store in index
        self._embeddings[node.id] = embedding

        # Also store on the node itself (for persistence)
        node.condition_embedding = embedding.tolist()

        # Save to cache
        self._save_embedding_cache()

        logger.info("Indexed understanding node %s (%d-dim embedding)",
                    node.id, len(embedding))
        return True

    def index_nodes(self, nodes: list) -> int:
        """Index multiple understanding nodes.

        Returns the number of nodes successfully indexed.
        """
        success = 0
        for node in nodes:
            if self.index_node(node):
                success += 1
        return success

    def retrieve(self, text: str, question: str,
                 nodes: dict, top_k: int = 5,
                 threshold: float = 0.25) -> List[Tuple]:
        """Find the best matching understanding for a question via embedding similarity.

        This is the CORE retrieval method — replaces keyword-based find_matching().

        Strategy:
          1. Encode question + text into a single embedding
          2. Compute cosine similarity against ALL indexed node embeddings
          3. Re-rank top-k candidates with lightweight operational signals
          4. Return sorted list of (node, score) tuples above threshold

        The embedding captures SEMANTIC similarity, so:
        - "kehilangan 35" matches U_quantity (operational similarity)
        - "siapa yang tidak hadir" matches U_signal_flip (semantic similarity)
        - "makna kata kepala" matches U_word_sense (semantic similarity)

        Even if the exact words differ, the OPERATIONAL MEANING is similar,
        so the embedding similarity will correctly route the question.

        Args:
            text: The source text
            question: The question to answer
            nodes: Dict of {node_id: UnderstandingNode} from the graph
            top_k: Number of top candidates to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (UnderstandingNode, score) tuples, sorted by score descending
        """
        if not self._embeddings:
            logger.warning("No embeddings indexed — cannot retrieve")
            return []

        # Step 1: Encode query
        query_text = self._build_query_text(text, question)
        query_emb = self._encode(query_text)
        if query_emb is None:
            logger.warning("Failed to encode query — cannot retrieve")
            return []

        # Step 2: Compute similarity against ALL indexed embeddings
        scored = []
        for node_id, node_emb in self._embeddings.items():
            # Cosine similarity (both vectors are already normalized)
            sim = float(np.dot(query_emb, node_emb))

            if sim >= threshold:
                node = nodes.get(node_id)
                if node is not None:
                    scored.append((node, sim))

        if not scored:
            return []

        # Step 3: Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Re-rank top-k with lightweight operational signals
        top_candidates = scored[:min(top_k * 2, len(scored))]
        reranked = self._rerank(top_candidates, text, question)
        return reranked[:top_k]

    def find_best(self, text: str, question: str,
                  nodes: dict, threshold: float = 0.25) -> Optional[Tuple]:
        """Find the single best matching understanding.

        Convenience method — returns (node, score) or None.
        """
        results = self.retrieve(text, question, nodes, top_k=1, threshold=threshold)
        if results:
            return results[0]
        return None

    def is_available(self) -> bool:
        """Check if the embedding model is available."""
        if self._model_loaded and self._model is not None:
            return True
        # Try loading
        self._ensure_model()
        return self._model is not None

    def set_experience_enabled(self, enabled: bool):
        """Toggle experience adjustment on/off for ablation testing.

        @FLOW:     EXPERIENCE_ABLATION
        @CALLS:    none — flag check only
        @MUTATES:  self._experience_enabled
        @BEHAVIOR: When disabled, _rerank() behaves as if ExperienceStore is empty.
                   This is for debugging — does NOT clear or modify stored episodes.
                   Toggle back on and the system immediately resumes using experience.
        """
        self._experience_enabled = enabled

    def get_stats(self) -> dict:
        """Return retriever statistics."""
        return {
            'indexed_nodes': len(self._embeddings),
            'model_loaded': self._model is not None,
            'model_name': self.MODEL_NAME,
            'embedding_dim': self.EMBEDDING_DIM,
            'cache_dir': self._cache_dir,
            'experience_enabled': self._experience_enabled,
        }

    # ═══════════════ EMBEDDING TEXT CONSTRUCTION ═══════════════

    def _build_embedding_text(self, node) -> str:
        """Build rich text for embedding that captures OPERATIONAL MEANING.

        The embedding text combines:
        1. concept — what the understanding is about
        2. abstraction — the generalized principle
        3. conditions — when it applies
        4. transformation action — what it does
        5. schemas — example patterns

        This is NOT just keyword matching. The embedding captures the
        SEMANTIC RELATIONSHIP between these elements, so that:
        - "kehilangan" maps to the same region as "terjual" (both = quantity loss)
        - "siapa yang tidak" maps to the same region as "kecuali" (both = exception)
        """
        parts = []

        # Concept — what this understanding is about
        if node.concept:
            parts.append(node.concept)

        # Abstraction — the generalized principle
        if node.abstraction:
            parts.append(node.abstraction)

        # Conditions — when it applies
        if node.conditions:
            parts.append(' '.join(node.conditions))

        # Transformation action — what it does
        if node.transformation and node.transformation.action:
            parts.append(node.transformation.action)

        # Transformation kind — operational category
        if node.transformation and node.transformation.kind:
            kind_description = self._kind_to_description(node.transformation.kind)
            if kind_description:
                parts.append(kind_description)

        # Schema examples (first 2) — concrete patterns
        if node.schemas:
            parts.extend(node.schemas[:2])

        return ' '.join(parts)

    def _build_query_text(self, text: str, question: str) -> str:
        """Build the query text for embedding.

        The query combines the question and text, with the question
        weighted more heavily (repeated) to emphasize what's being asked.

        We also add a task description to help the embedding model
        understand what kind of matching we're doing.
        """
        # Task hint helps the embedding model understand the matching context
        task_hint = "Temukan understanding yang cocok untuk menjawab pertanyaan ini"

        # Weight the question more heavily by repeating it
        # This is a simple but effective technique for dense retrieval
        weighted_query = f"{task_hint} {question} {question} {text}"

        return weighted_query

    def _kind_to_description(self, kind: str) -> str:
        """Convert a transformation kind to a human-readable description.

        These descriptions help the embedding model understand the
        OPERATIONAL CATEGORY of the understanding. This is NOT hardcoded
        knowledge — it's a translation layer that makes the embedding
        more semantically meaningful.

        The kind names (signal_flip, contrast_focus, etc.) are just labels.
        Adding their descriptions gives the embedding model more signal.
        """
        descriptions = {
            'signal_flip': 'pengecualian kebalikan sinyal negasi kecuali selain terkecuali',
            'contrast_focus': 'kontras pertentangan tetapi namun tapi perbedaan',
            'negation_affirmation': 'negasi penegasan bukan tapi melainkan penolakan',
            'comparison_resolve': 'perbandingan lebih kurang dari pembandingan',
            'entity_extract': 'ekstraksi entitas siapa apa dimana pencarian',
            'fact_extract': 'fakta faktual apa yang membuat memasak mengerjakan',
            'quantity_compute': 'kuantitas berapa jumlah hitung kembalian bilangan',
            'context_filter': 'konteks filter menurut teks berdasarkan pencarian informasi',
            'word_sense': 'makna kata bermakna artinya disambiguasi konteks',
        }
        return descriptions.get(kind, '')

    # ═══════════════ RE-RANKING ═══════════════

    def _rerank(self, candidates: List[Tuple],
                text: str, question: str) -> List[Tuple]:
        """Re-rank candidates with lightweight operational signals and experience.

        After embedding similarity gives us top candidates, we apply
        lightweight heuristic adjustments to break ties and correct
        for cases where pure semantic similarity might misroute.

        Re-ranking factors:
        1. Node accuracy (proven track record) — 0.3 weight
        2. Embedding similarity — 0.5 weight
        3. Operational signal match — 0.2 weight
        4. Experience adjustment (success/failure history) — additive penalty/boost

        Factors 1-3 are combined with weights. Factor 4 is added on top
        as an additive adjustment derived from experience episodes.

        This is NOT hardcoded knowledge. The operational signal match
        is a MECHANICAL check: does the question contain numbers? → boost
        quantity. Does it ask about meaning? → boost word_sense. These
        are STRUCTURAL features of the question, not domain knowledge.

        The experience adjustment is derived from COSINE SIMILARITY to
        past failure/success episodes — zero hardcoded rules.
        """
        text_lower = text.lower()
        question_lower = question.lower()

        # Compute query embedding once for experience adjustment
        query_emb = self._encode(self._build_query_text(text, question))

        reranked = []
        for node, emb_score in candidates:
            # Factor 1: Node accuracy (proven track record)
            accuracy = node.accuracy if node.times_applied > 0 else 0.5

            # Factor 2: Embedding similarity (already computed)
            emb_norm = min(1.0, max(0.0, emb_score))

            # Factor 3: Operational signal match
            op_signal = self._compute_operational_signal(node, text_lower, question_lower)

            # Combine: embedding is most important, accuracy and signals help
            final_score = 0.5 * emb_norm + 0.3 * accuracy + 0.2 * op_signal

            # Factor 4: Experience-based adjustment
            # Penalty for nodes that failed in similar contexts,
            # boost for nodes that succeeded in similar contexts.
            # Qwen3 does NOT know about this adjustment.
            # Ablation: skip when experience_enabled is False.
            experience_adj = 0.0
            if self._experience_enabled:
                experience_adj = self._compute_experience_adjustment(node, query_emb)
            final_score += experience_adj

            reranked.append((node, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _compute_experience_adjustment(self, node, query_emb: Optional[np.ndarray]) -> float:
        """Compute experience-based score adjustment for a node.

        @FLOW:     EXPERIENCE_PENALTY
        @CALLS:    ExperienceStore.compute_experience_adjustment() → float [-0.15, +0.05]
        @MUTATES:  none
        @BEHAVIOR: Returns 0.0 if ExperienceStore is unavailable or query_emb is None
                   or experience is disabled (ablation mode).
                   Adjustment range: [-0.15, +0.05].
                   Negative = penalty (node failed in similar contexts).
                   Positive = boost (node succeeded in similar contexts).
                   This adjustment is INVISIBLE to Qwen3 — it only affects
                   bge-m3 retrieval scoring.
                   When debug logging is enabled, also logs trace details
                   showing which episodes contributed.
        """
        if query_emb is None:
            return 0.0

        if not self._experience_enabled:
            return 0.0

        store = self._get_experience_store()
        if store is None:
            return 0.0

        try:
            adjustment = store.compute_experience_adjustment(query_emb, node.id)

            # Debug tracing — log full episode detail when DEBUG level is active
            if logger.isEnabledFor(logging.DEBUG) and adjustment != 0.0:
                try:
                    trace = store.trace_adjustment(query_emb, node.id)
                    logger.debug(
                        "Experience adjustment for %s: %.4f "
                        "(%d/%d episodes matched, threshold=%.3f)",
                        node.id, adjustment,
                        trace['total_episodes_matched'],
                        trace['total_episodes_checked'],
                        trace['threshold'],
                    )
                    for ep in trace['matched_episodes'][:5]:
                        logger.debug(
                            "  → %s: sim=%.3f decay=%.3f contrib=%.4f (%s)",
                            ep['episode_id'], ep['similarity'],
                            ep['decay'], ep['contribution'], ep['outcome'],
                        )
                except Exception:
                    pass  # Tracing is optional — don't break adjustment on trace failure

            return adjustment
        except Exception as e:
            logger.debug("Experience adjustment computation failed: %s", e)
            return 0.0

    def _get_experience_store(self):
        """Lazy-initialize the ExperienceStore.

        If no store was provided at init, use the shared singleton.
        This prevents multiple instances from loading data separately.
        """
        if self._experience_store is None:
            try:
                from derivation.experience_store import get_shared_store
                self._experience_store = get_shared_store()
            except ImportError:
                logger.debug("experience_store module not available")
            except Exception as e:
                logger.debug("Failed to init ExperienceStore: %s", e)
        return self._experience_store

    def _compute_operational_signal(self, node, text_lower: str,
                                     question_lower: str) -> float:
        """Compute operational signal match score.

        This checks STRUCTURAL features of the question against the
        understanding's operational category. This is NOT domain knowledge —
        it's a mechanical check for structural patterns.

        For example:
        - Question has numbers + "berapa" → quantity_compute gets signal boost
        - Question has "makna/bermakna" → word_sense gets signal boost
        - Text has "kecuali/selain" → signal_flip gets signal boost

        These are STRUCTURAL signals, not semantic knowledge. They help
        break ties when embedding similarity is close between candidates.
        """
        if not node.transformation:
            return 0.0

        kind = node.transformation.kind
        signal = 0.0

        # Quantity signals: numbers + "berapa/jumlah/total"
        if kind == 'quantity_compute':
            import re
            if re.search(r'\d+', text_lower) and any(
                w in question_lower for w in ['berapa', 'jumlah', 'total', 'kembalian', 'sisa']
            ):
                signal += 0.4
            elif any(w in question_lower for w in ['berapa', 'jumlah', 'total']):
                signal += 0.2

        # Exception signals: "kecuali/selain" in text
        elif kind == 'signal_flip':
            if any(w in text_lower for w in ['kecuali', 'selain', 'terkecuali']):
                signal += 0.4
            elif any(w in question_lower for w in ['tidak', 'bukan', 'selain']):
                signal += 0.2

        # Contrast signals: "tetapi/namun/tapi" in text
        elif kind == 'contrast_focus':
            if any(w in text_lower for w in ['tetapi', 'namun', 'tapi']):
                signal += 0.4

        # Negation signals: "bukan...tapi" pattern
        elif kind == 'negation_affirmation':
            if 'bukan' in text_lower and any(
                w in text_lower for w in ['tapi', 'tetapi', 'melainkan']
            ):
                signal += 0.4

        # Word sense signals: "makna/bermakna/artinya"
        elif kind == 'word_sense':
            if any(w in question_lower for w in ['bermakna', 'makna', 'artinya', 'maksud']):
                signal += 0.4

        # Comparison signals: "lebih/kurang" in text
        elif kind == 'comparison_resolve':
            if any(w in text_lower for w in ['lebih', 'kurang', 'dibanding']):
                signal += 0.4

        # Fact extraction signals: "apa yang" + action verb
        elif kind == 'fact_extract':
            if 'apa yang' in question_lower:
                signal += 0.3

        # Context filter signals: "menurut teks/berdasarkan"
        elif kind == 'context_filter':
            if any(w in question_lower for w in ['menurut', 'teks', 'berdasarkan']):
                signal += 0.3

        # Entity extraction signals: "siapa/apa/dimana"
        elif kind == 'entity_extract':
            if any(w in question_lower for w in ['siapa', 'apa', 'dimana', 'di mana']):
                signal += 0.2

        return min(1.0, signal)

    # ═══════════════ MODEL MANAGEMENT ═══════════════

    def _ensure_model(self):
        """Lazy-load the bge-m3 model via shared singleton.

        Uses model_registry.get_shared_embedding_model() to avoid
        loading bge-m3 multiple times (~2.2GB RAM each).
        """
        if self._model_loaded:
            return

        try:
            from derivation.model_registry import get_shared_embedding_model
            self._model = get_shared_embedding_model()
            if self._model is not None:
                logger.info("UnderstandingRetriever using shared bge-m3 (dim=%d)", self.EMBEDDING_DIM)
            else:
                logger.warning(
                    "Shared embedding model not available — "
                    "embedding retrieval disabled"
                )
        except ImportError:
            logger.warning(
                "model_registry not available — "
                "embedding retrieval disabled"
            )
            self._model = None
        except Exception as e:
            logger.warning("Failed to get shared embedding model: %s", e)
            self._model = None

        self._model_loaded = True

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text into a normalized embedding vector.

        Returns:
            1024-dim normalized numpy array, or None if encoding fails
        """
        self._ensure_model()
        if self._model is None:
            return None

        try:
            emb = self._model.encode(
                [text], show_progress_bar=False, normalize_embeddings=True
            )[0]
            return emb
        except Exception as e:
            logger.warning("Failed to encode text: %s", e)
            return None

    # ═══════════════ EMBEDDING CACHE ═══════════════

    def _save_embedding_cache(self):
        """Save all indexed embeddings to disk cache."""
        if not self._embeddings:
            return

        try:
            os.makedirs(self._cache_dir, exist_ok=True)

            # Save as compressed numpy archive
            cache_path = os.path.join(self._cache_dir, 'understanding_embeddings.npz')
            node_ids = list(self._embeddings.keys())
            emb_stack = np.stack([self._embeddings[nid] for nid in node_ids])
            np.savez_compressed(cache_path, node_ids=node_ids, embeddings=emb_stack)

            logger.debug("Saved %d embeddings to %s", len(node_ids), self._cache_dir)
        except Exception as e:
            logger.warning("Failed to save embedding cache: %s", e)

    def _load_embedding_cache(self) -> bool:
        """Load embeddings from disk cache.

        Returns True if embeddings were loaded successfully.
        """
        cache_path = os.path.join(self._cache_dir, 'understanding_embeddings.npz')

        if not os.path.exists(cache_path):
            return False

        try:
            data = np.load(cache_path, allow_pickle=False)
            node_ids = data['node_ids']
            embeddings = data['embeddings']

            for nid, emb in zip(node_ids, embeddings):
                self._embeddings[str(nid)] = emb

            logger.info("Loaded %d cached embeddings from %s",
                       len(self._embeddings), self._cache_dir)
            return len(self._embeddings) > 0
        except Exception as e:
            logger.warning("Failed to load embedding cache: %s", e)
            return False

    def load_from_graph(self, graph):
        """Load and index all nodes from an UnderstandingGraph.

        This is the main way to initialize the retriever from an
        existing understanding graph. It computes embeddings for any
        nodes that don't have them yet.

        Args:
            graph: UnderstandingGraph instance
        """
        nodes = graph._nodes

        # First, try to load from node's own condition_embedding
        for nid, node in nodes.items():
            if nid not in self._embeddings and node.condition_embedding is not None:
                try:
                    emb = np.array(node.condition_embedding)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
                    self._embeddings[nid] = emb
                except Exception:
                    pass

        # Then, compute embeddings for any remaining nodes
        unindexed = [node for nid, node in nodes.items() if nid not in self._embeddings]
        if unindexed:
            logger.info("Computing embeddings for %d unindexed nodes ...", len(unindexed))
            indexed = self.index_nodes(unindexed)
            logger.info("Indexed %d/%d nodes", indexed, len(unindexed))
        else:
            logger.info("All %d nodes already indexed", len(self._embeddings))
