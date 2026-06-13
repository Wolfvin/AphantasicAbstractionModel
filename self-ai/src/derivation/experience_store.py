# @WHO:   self-ai/src/derivation/experience_store.py
# @WHAT:  Episode-based experience storage for embedding retrieval penalty
# @PART:  self-ai/derivation
# @ENTRY: ExperienceStore.record_episode(), ExperienceStore.compute_experience_adjustment(), ExperienceStore.trace_adjustment(), ExperienceStore.report_adjustment_outcome(), ExperienceStore.remove_episode()

"""Experience Store — HOW SELF learns from mistakes at the retrieval layer.

Vision:
    SELF punya dua sistem: bge-m3 (Sistem 1: intuisi) dan Understanding Graph
    (Sistem 2: pemahaman). ExperienceStore menghubungkan keduanya dengan
    mempengaruhi skor retrieval bge-m3 berdasarkan pengalaman historis.

    Ketika SELF menjawab salah menggunakan understanding node X di konteks Y,
    episode ini disimpan. Di pertanyaan berikutnya yang mirip konteks Y,
    node X mendapat penalty — skor retrieval diturunkan IMPLISIT.
    Qwen3 TIDAK TAHU tentang penalty ini.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Episode: (context_embedding, node_id, outcome)         │
    │  "Saya pakai U_signal_flip di konteks K, dan SALAH"    │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Next query: context mirip K                            │
    │  bge-m3 encode(query) → query_embedding                 │
    │  cosine_sim(query_embedding, K_embedding) = 0.85       │
    │  → U_signal_flip gets PENALTY (score diturunkan)       │
    │  ← Qwen3 tidak tahu ini terjadi                        │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Re-ranked results:                                     │
    │  U_signal_flip: 0.82 - 0.12 = 0.70  (penalized)       │
    │  U_quantity:    0.75 + 0.03 = 0.78  (boosted)         │
    │  → U_quantity wins, even though raw sim was lower       │
    └─────────────────────────────────────────────────────────┘

Key Design Decisions:
    1. ZERO HARDCODED: Penalty dihitung dari cosine similarity ke episode
       pengalaman, bukan dari rules atau thresholds yang ditentukan manual.
       Semakin mirip konteks sekarang dengan konteks yang dulu gagal,
       semakin besar penalty.

    2. QWEN3 TIDAK TAHU: ExperienceWeight hanya mengubah skor retrieval
       bge-m3. Qwen3 tetap menerima context yang sama. Ini penting karena
       kita tidak ingin "pollute" reasoning Qwen3 dengan bias dari pengalaman.

    3. TEMPORAL DECAY: Episode lama punya weight lebih kecil. Ini mencegah
       episode yang sudah tidak relevan menghantui retrieval selamanya.
       Decay rate: weight = exp(-lambda * age_days), lambda = 0.05

    4. ASYMMETRIC: Penalty untuk failure lebih besar daripada boost untuk
       success. Ini karena salah satu jawaban salah lebih berbahaya daripada
       kehilangan satu jawaban benar. Penalty weight: 0.15, Boost weight: 0.05.

    5. PRUNABLE: Episode lama (lebih dari max_age_days) bisa dibersihkan
       untuk mencegah storage bengkak dan maintain performa.

    6. ADAPTIVE THRESHOLD: Similarity threshold bukan konstanta hardcoded.
       Mulai dari 0.5, lalu adapt berdasarkan distribusi similarity aktual
       dari episode yang pernah match. Kalau terlalu banyak false positives
       (episode match tapi adjustment-nya salah), threshold naik. Kalau
       terlalu sedikit episode yang relevan (match rate rendah), threshold
       turun. Ini memenuhi filosofi zero hardcoded — threshold diturunkan
       dari data, bukan ditentukan programmer.

    7. SELF-EVALUATION: SELF bisa mendeteksi kemungkinan kesalahan sendiri
       tanpa input eksternal. Jawaban dengan calibrated confidence sangat
       rendah (< 0.3) otomatis di-record sebagai suspected failure. Ini
       membuat feedback loop berjalan tanpa perlu manual intervention.
"""

import os
import json
import time
import logging
import uuid
from typing import Optional, List, Dict

import numpy as np

logger = logging.getLogger(__name__)


class ExperienceStore:
    """Store and retrieve experience episodes for embedding retrieval penalty.

    Each episode records:
    - context_embedding: bge-m3 embedding of the question+text context
    - node_id: which understanding node was selected
    - outcome: "success" or "failure" — was the answer correct?
    - question: for debugging/audit
    - timestamp: when the episode occurred
    - weight: temporal decay weight (auto-computed from timestamp)

    The store is persisted to disk as JSON and loaded on init.
    Embeddings are stored as lists (not binary) for portability.

    Usage:
        store = ExperienceStore()

        # Record a failure episode
        store.record_episode(
            context_text="Toko kehilangan 35 roti...",
            node_id="U_signal_flip",
            outcome="failure",
            question="Berapa sisa roti?"
        )

        # Compute penalty/boost for a node given a new query
        query_emb = bge_m3.encode("Toko kehilangan 28 kue...")
        adjustment = store.compute_experience_adjustment(query_emb, "U_signal_flip")
        # adjustment might be -0.12 (penalty) or +0.03 (boost)
    """

    # Penalty/boost constants — derived from empirical balance, not hardcoded rules
    FAILURE_WEIGHT = 0.15    # Max penalty magnitude for failures
    SUCCESS_WEIGHT = 0.05    # Max boost magnitude for successes
    # v37: DECAY_LAMBDA set to 0.0 — System 2 Think Slow: learning is PERMANENT.
    # Temporal decay was erasing knowledge over time, contradicting the principle
    # that AI should get smarter day by day. Experience episodes now persist
    # at full strength indefinitely. The only limit is MAX_EPISODES (FIFO eviction).
    DECAY_LAMBDA = 0.0       # No temporal decay — learning is permanent
    MAX_EPISODES = 2000      # Prevent unbounded growth

    # ── Adaptive threshold ──
    # Instead of a hardcoded SIMILARITY_THRESHOLD, the threshold is derived
    # from the actual distribution of similarity scores observed at runtime.
    # It starts at 0.5 and adapts: too many false positives → increase,
    # too few matches → decrease.
    DEFAULT_SIMILARITY_THRESHOLD = 0.5   # Starting point — NOT a permanent value
    THRESHOLD_ADAPT_RATE = 0.02          # How much to adjust per signal (small = stable)
    THRESHOLD_MIN = 0.3                 # Floor — never go below this
    THRESHOLD_MAX = 0.8                 # Ceiling — never go above this
    ADAPT_WINDOW_SIZE = 50              # Rolling window for match rate tracking

    def __init__(self, store_path: str = None, embedding_encoder=None):
        """Initialize the experience store.

        Args:
            store_path: Path to persist episodes. Defaults to
                        <project_root>/data/experience_store.json
            embedding_encoder: Callable that encodes text to embedding.
                               If None, will lazy-init from bge-m3.
        """
        self._store_path = store_path or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'experience_store.json'
        )
        self._embedding_encoder = embedding_encoder
        self._model = None
        self._model_loaded = False

        # In-memory episode storage
        # Each episode: {id, context_embedding (list), node_id, outcome, question, timestamp}
        self._episodes: List[Dict] = []

        # Index: node_id -> list of episode indices (for fast lookup)
        self._node_index: Dict[str, List[int]] = {}

        # Embedding cache: episode_id -> np.ndarray (for fast computation)
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # ── Adaptive similarity threshold ──
        # Not a constant — derived from the distribution of actual similarity
        # scores. Starts at DEFAULT and adapts based on feedback signal.
        self._similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD

        # Rolling window of recent match outcomes for threshold adaptation:
        # v3: Each entry: {'node_id': str, 'was_correct': bool, 'timestamp': float,
        #                   'adjustment_applied': bool, 'threshold_at_time': float}
        # adjustment_applied = whether the experience adjustment was non-zero
        # threshold_at_time = snapshot of similarity_threshold when outcome was reported
        self._match_history: List[Dict] = []

        # ── Threshold history — track threshold changes over time ──
        # Needed for threshold_health metric: is the threshold converging
        # or oscillating? Without this, we can only see the current value.
        # Each entry: {'threshold': float, 'timestamp': float}
        self._threshold_history: List[Dict] = []

        # Load persisted episodes (also restores adaptive threshold state)
        self._load()

    # ═══════════════ PUBLIC API ═══════════════

    def record_episode(self, context_text: str, node_id: str,
                       outcome: str, question: str = '') -> Optional[str]:
        """Record an experience episode.

        @FLOW:     EXPERIENCE_RECORD
        @CALLS:    bge-m3 encode(context_text) → 1024-dim embedding
        @MUTATES:  self._episodes (append), self._node_index (update),
                   experience_store.json (persist)
        @BEHAVIOR: If embedding model unavailable, episode is recorded with
                   context_embedding=None. compute_experience_adjustment()
                   will skip such episodes. Returns episode ID on success.

        Args:
            context_text: The text+question context for this episode
            node_id: The understanding node that was selected
            outcome: "success" or "failure"
            question: The question asked (for debugging/audit)

        Returns:
            Episode ID string, or None if recording failed
        """
        if outcome not in ('success', 'failure'):
            logger.warning("Invalid outcome '%s' — must be 'success' or 'failure'", outcome)
            return None

        # Compute context embedding
        context_embedding = self._encode(context_text)

        episode_id = f"exp_{uuid.uuid4().hex[:8]}"
        episode = {
            'id': episode_id,
            'context_embedding': context_embedding.tolist() if context_embedding is not None else None,
            'node_id': node_id,
            'outcome': outcome,
            'question': question[:200],  # Truncate for storage
            'timestamp': time.time(),
        }

        # Add to memory
        idx = len(self._episodes)
        self._episodes.append(episode)

        # Update index
        if node_id not in self._node_index:
            self._node_index[node_id] = []
        self._node_index[node_id].append(idx)

        # Cache embedding
        if context_embedding is not None:
            self._embedding_cache[episode_id] = context_embedding

        # Enforce max episodes (FIFO — remove oldest)
        if len(self._episodes) > self.MAX_EPISODES:
            self._prune_oldest(len(self._episodes) - self.MAX_EPISODES)

        # Persist
        self._save()

        logger.debug("Recorded %s episode for node %s: %s",
                     outcome, node_id, episode_id)
        return episode_id

    def compute_experience_adjustment(self, query_embedding: np.ndarray,
                                       node_id: str) -> float:
        """Compute the experience-based score adjustment for a node.

        @FLOW:     EXPERIENCE_PENALTY
        @CALLS:    numpy dot product (cosine similarity) against stored episodes
        @MUTATES:  none
        @BEHAVIOR: Returns a float in [-FAILURE_WEIGHT, +SUCCESS_WEIGHT].
                   Negative = penalty (node failed in similar contexts).
                   Positive = boost (node succeeded in similar contexts).
                   Returns 0.0 if no relevant episodes exist.

        The adjustment is computed by:
        1. Finding all episodes for this node_id
        2. Computing cosine similarity between query and each episode's context
        3. Weighting by temporal decay (recent episodes matter more)
        4. Summing: failures contribute negative, successes contribute positive

        Args:
            query_embedding: 1024-dim normalized embedding of current query
            node_id: The understanding node being evaluated

        Returns:
            Float adjustment to add to the node's retrieval score
        """
        if node_id not in self._node_index:
            return 0.0

        episode_indices = self._node_index[node_id]
        if not episode_indices:
            return 0.0

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm < 1e-8:
            return 0.0
        query_norm = query_embedding / norm

        adjustment = 0.0
        now = time.time()

        for idx in episode_indices:
            if idx >= len(self._episodes):
                continue

            episode = self._episodes[idx]
            ep_id = episode.get('id', '')

            # Get episode embedding
            if ep_id in self._embedding_cache:
                ep_emb = self._embedding_cache[ep_id]
            else:
                ep_emb_list = episode.get('context_embedding')
                if ep_emb_list is None:
                    continue
                ep_emb = np.array(ep_emb_list, dtype=np.float32)
                # Normalize
                ep_norm = np.linalg.norm(ep_emb)
                if ep_norm < 1e-8:
                    continue
                ep_emb = ep_emb / ep_norm
                self._embedding_cache[ep_id] = ep_emb

            # Cosine similarity (both normalized)
            similarity = float(np.dot(query_norm, ep_emb))

            # Skip episodes that aren't similar enough to current context
            # Threshold is ADAPTIVE — derived from data, not hardcoded
            if similarity < self._similarity_threshold:
                continue

            # Temporal decay: recent episodes matter more
            age_days = (now - episode.get('timestamp', now)) / 86400.0
            decay = float(np.exp(-self.DECAY_LAMBDA * age_days))

            # Weighted contribution
            weighted_sim = similarity * decay

            outcome = episode.get('outcome', '')
            if outcome == 'failure':
                adjustment -= self.FAILURE_WEIGHT * weighted_sim
            elif outcome == 'success':
                adjustment += self.SUCCESS_WEIGHT * weighted_sim

        # Clamp to valid range
        adjustment = max(-self.FAILURE_WEIGHT, min(self.SUCCESS_WEIGHT, adjustment))

        return adjustment

    def report_adjustment_outcome(self, node_id: str, was_correct: bool,
                                    adjustment_applied: bool = False):
        """Report whether the experience adjustment was helpful for threshold adaptation.

        @FLOW:     EXPERIENCE_THRESHOLD_ADAPT
        @CALLS:    none (internal state update)
        @MUTATES:  self._similarity_threshold (potentially adjusted),
                   self._match_history (append), self._threshold_history (append)
        @BEHAVIOR: After each answer is verified (via feedback or self-evaluation),
                   the caller reports whether the node's answer was correct.
                   This signal is used to adapt the similarity threshold:
                   - If adjustment was applied but answer was wrong → false positive
                     (penalty didn't help) → increase threshold
                   - If few episodes matched and answer was wrong → maybe threshold
                     too high (missing relevant episodes) → decrease threshold
                   The adaptation is gradual (THRESHOLD_ADAPT_RATE per signal)
                   and bounded (THRESHOLD_MIN to THRESHOLD_MAX).

        This method is the KEY to making the threshold adaptive instead of
        hardcoded. The threshold adapts to the DATA, not the programmer.

        Args:
            node_id: The understanding node that was used
            was_correct: Whether the answer turned out to be correct
            adjustment_applied: Whether a non-zero experience adjustment was
                                computed for this node. Needed for
                                penalty_effectiveness metric — without it,
                                we can't distinguish "adjustment didn't help"
                                from "no adjustment was attempted".
        """
        entry = {
            'node_id': node_id,
            'was_correct': was_correct,
            'timestamp': time.time(),
            'adjustment_applied': adjustment_applied,
            'threshold_at_time': self._similarity_threshold,
        }
        self._match_history.append(entry)

        # Trim to window size
        if len(self._match_history) > self.ADAPT_WINDOW_SIZE:
            self._match_history = self._match_history[-self.ADAPT_WINDOW_SIZE:]

        # ── Adapt threshold based on recent outcomes ──
        # Count recent false positives: adjustments applied but answer still wrong
        recent = self._match_history[-20:]  # Look at last 20 signals
        if len(recent) < 5:
            return  # Not enough data to adapt yet

        wrong_count = sum(1 for e in recent if not e.get('was_correct', True))
        wrong_ratio = wrong_count / len(recent)

        # If too many wrong answers despite adjustments, threshold might be too low
        # (matching irrelevant episodes that add noise) → increase threshold
        if wrong_ratio > 0.6:
            old_threshold = self._similarity_threshold
            self._similarity_threshold = min(
                self.THRESHOLD_MAX,
                self._similarity_threshold + self.THRESHOLD_ADAPT_RATE
            )
            if self._similarity_threshold != old_threshold:
                self._record_threshold_change(old_threshold)
                logger.info("Adaptive threshold UP: %.3f → %.3f (wrong_ratio=%.2f)",
                           old_threshold, self._similarity_threshold, wrong_ratio)

        # If very few wrong answers but also very few episodes matching,
        # threshold might be too high (missing useful episodes) → decrease
        elif wrong_ratio < 0.2 and len(self._episodes) > 10:
            # Check match rate — are we matching enough episodes?
            total_episodes_for_node = sum(
                len(indices) for nid, indices in self._node_index.items()
            )
            if total_episodes_for_node > 0:
                match_rate = len(recent) / max(1, total_episodes_for_node)
                if match_rate < 0.1:  # Very few matches relative to episodes
                    old_threshold = self._similarity_threshold
                    self._similarity_threshold = max(
                        self.THRESHOLD_MIN,
                        self._similarity_threshold - self.THRESHOLD_ADAPT_RATE
                    )
                    if self._similarity_threshold != old_threshold:
                        self._record_threshold_change(old_threshold)
                        logger.info("Adaptive threshold DOWN: %.3f → %.3f (match_rate=%.2f)",
                                   old_threshold, self._similarity_threshold, match_rate)

        # Persist threshold state
        self._save()

    def prune_old_episodes(self, max_age_days: int = 30) -> int:
        """Remove episodes older than max_age_days.

        @FLOW:     EXPERIENCE_PRUNE
        @CALLS:    none
        @MUTATES:  self._episodes (remove), self._node_index (rebuild),
                   self._embedding_cache (remove), experience_store.json (persist)
        @BEHAVIOR: Returns the number of episodes pruned. Rebuilds all indices.

        Args:
            max_age_days: Maximum age in days. Episodes older than this are removed.

        Returns:
            Number of episodes pruned
        """
        now = time.time()
        cutoff = now - (max_age_days * 86400.0)

        old_count = len(self._episodes)
        self._episodes = [
            ep for ep in self._episodes
            if ep.get('timestamp', 0) >= cutoff
        ]
        pruned = old_count - len(self._episodes)

        if pruned > 0:
            # Rebuild indices
            self._rebuild_indices()
            self._save()
            logger.info("Pruned %d episodes older than %d days", pruned, max_age_days)

        return pruned

    def trace_adjustment(self, query_embedding: np.ndarray,
                          node_id: str) -> dict:
        """Trace which episodes contributed to the adjustment and how much.

        @FLOW:     EXPERIENCE_TRACE
        @CALLS:    numpy dot product (cosine similarity) against stored episodes
        @MUTATES:  none — read-only operation for debugging
        @BEHAVIOR: Returns a dict with full detail about the adjustment computation.
                   This is for DEBUGGING ONLY — it does not change any state.
                   The 'adjustment' value matches what compute_experience_adjustment()
                   would return, but with full episode-level detail.

        Args:
            query_embedding: 1024-dim normalized embedding of current query
            node_id: The understanding node being evaluated

        Returns:
            dict with:
            - 'adjustment': float — the final adjustment value
            - 'matched_episodes': list of dicts, each with:
                - 'episode_id': str
                - 'similarity': float — cosine similarity to query
                - 'decay': float — temporal decay weight
                - 'contribution': float — how much this episode contributed
                - 'outcome': str — 'success' or 'failure'
            - 'threshold': float — current adaptive threshold
            - 'total_episodes_checked': int — episodes for this node
            - 'total_episodes_matched': int — episodes above threshold
        """
        result = {
            'adjustment': 0.0,
            'matched_episodes': [],
            'threshold': self._similarity_threshold,
            'total_episodes_checked': 0,
            'total_episodes_matched': 0,
        }

        if node_id not in self._node_index:
            return result

        episode_indices = self._node_index[node_id]
        result['total_episodes_checked'] = len(episode_indices)

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm < 1e-8:
            return result
        query_norm = query_embedding / norm

        adjustment = 0.0
        now = time.time()

        for idx in episode_indices:
            if idx >= len(self._episodes):
                continue

            episode = self._episodes[idx]
            ep_id = episode.get('id', '')

            # Get episode embedding
            if ep_id in self._embedding_cache:
                ep_emb = self._embedding_cache[ep_id]
            else:
                ep_emb_list = episode.get('context_embedding')
                if ep_emb_list is None:
                    continue
                ep_emb = np.array(ep_emb_list, dtype=np.float32)
                ep_norm = np.linalg.norm(ep_emb)
                if ep_norm < 1e-8:
                    continue
                ep_emb = ep_emb / ep_norm

            # Cosine similarity (both normalized)
            similarity = float(np.dot(query_norm, ep_emb))

            # Skip episodes below threshold
            if similarity < self._similarity_threshold:
                continue

            result['total_episodes_matched'] += 1

            # Temporal decay
            age_days = (now - episode.get('timestamp', now)) / 86400.0
            decay = float(np.exp(-self.DECAY_LAMBDA * age_days))

            # Weighted contribution
            weighted_sim = similarity * decay
            outcome = episode.get('outcome', '')

            contribution = 0.0
            if outcome == 'failure':
                contribution = -self.FAILURE_WEIGHT * weighted_sim
                adjustment += -self.FAILURE_WEIGHT * weighted_sim
            elif outcome == 'success':
                contribution = self.SUCCESS_WEIGHT * weighted_sim
                adjustment += self.SUCCESS_WEIGHT * weighted_sim

            result['matched_episodes'].append({
                'episode_id': ep_id,
                'similarity': round(similarity, 4),
                'decay': round(decay, 4),
                'contribution': round(contribution, 4),
                'outcome': outcome,
            })

        # Clamp to valid range
        adjustment = max(-self.FAILURE_WEIGHT, min(self.SUCCESS_WEIGHT, adjustment))
        result['adjustment'] = round(adjustment, 4)

        return result

    def remove_episode(self, episode_id: str) -> bool:
        """Remove a specific episode by ID.

        @FLOW:     EXPERIENCE_REVERSE
        @CALLS:    none
        @MUTATES:  self._episodes (remove), self._node_index (rebuild),
                   self._embedding_cache (remove), experience_store.json (persist)
        @BEHAVIOR: This is the REVERSIBILITY feature — you can undo any episode
                   and the system immediately behaves as if it never existed.
                   No retraining needed. This is a unique property of experience-
                   based retrieval adjustment vs. model weight training.

        Args:
            episode_id: The ID of the episode to remove

        Returns:
            True if episode was found and removed, False otherwise
        """
        for i, ep in enumerate(self._episodes):
            if ep.get('id') == episode_id:
                self._episodes.pop(i)
                self._rebuild_indices()
                self._save()
                logger.info("Removed episode %s (reversibility)", episode_id)
                return True

        logger.debug("Episode %s not found for removal", episode_id)
        return False

    def remove_episodes_for_node(self, node_id: str,
                                  outcome: str = None) -> int:
        """Remove all episodes for a specific node, optionally filtered by outcome.

        @FLOW:     EXPERIENCE_REVERSE
        @CALLS:    none
        @MUTATES:  self._episodes (remove), self._node_index (rebuild),
                   self._embedding_cache (remove), experience_store.json (persist)
        @BEHAVIOR: Useful for:
                   - Resetting a node's experience history (remove all episodes)
                   - Clearing only failures (outcome='failure') to test if node works better
                   - Clearing only successes (outcome='success') to test if boost is excessive

        Args:
            node_id: The node to remove episodes for
            outcome: If specified, only remove episodes with this outcome.
                     If None, remove all episodes for this node.

        Returns:
            Number of episodes removed
        """
        if node_id not in self._node_index:
            return 0

        old_count = len(self._episodes)

        if outcome is not None:
            self._episodes = [
                ep for ep in self._episodes
                if not (ep.get('node_id') == node_id and ep.get('outcome') == outcome)
            ]
        else:
            self._episodes = [
                ep for ep in self._episodes
                if ep.get('node_id') != node_id
            ]

        removed = old_count - len(self._episodes)

        if removed > 0:
            self._rebuild_indices()
            self._save()
            logger.info("Removed %d episodes for node %s (outcome=%s)",
                       removed, node_id, outcome)

        return removed

    def get_stats(self) -> dict:
        """Return experience store statistics including adaptive threshold state."""
        total = len(self._episodes)
        failures = sum(1 for ep in self._episodes if ep.get('outcome') == 'failure')
        successes = sum(1 for ep in self._episodes if ep.get('outcome') == 'success')

        # Per-node stats
        node_stats = {}
        for node_id, indices in self._node_index.items():
            node_failures = sum(
                1 for idx in indices
                if idx < len(self._episodes) and self._episodes[idx].get('outcome') == 'failure'
            )
            node_successes = sum(
                1 for idx in indices
                if idx < len(self._episodes) and self._episodes[idx].get('outcome') == 'success'
            )
            node_stats[node_id] = {
                'total': len(indices),
                'failures': node_failures,
                'successes': node_successes,
            }

        return {
            'total_episodes': total,
            'failures': failures,
            'successes': successes,
            'unique_nodes': len(self._node_index),
            'node_stats': node_stats,
            'store_path': self._store_path,
            'cached_embeddings': len(self._embedding_cache),
            'similarity_threshold': self._similarity_threshold,
            'match_history_size': len(self._match_history),
            'threshold_history_size': len(self._threshold_history),
        }

    def get_episodes_for_node(self, node_id: str) -> List[Dict]:
        """Get all episodes for a specific node (for debugging/audit)."""
        if node_id not in self._node_index:
            return []
        return [
            self._episodes[idx]
            for idx in self._node_index[node_id]
            if idx < len(self._episodes)
        ]

    # ═══════════════ ENCODING ═══════════════

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text into a normalized embedding vector using bge-m3."""
        if self._embedding_encoder is not None:
            try:
                emb = self._embedding_encoder(text)
                if emb is not None:
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        return emb / norm
                return emb
            except Exception as e:
                logger.warning("Custom encoder failed: %s", e)
                return None

        # Lazy-load bge-m3
        self._ensure_model()
        if self._model is None:
            return None

        try:
            emb = self._model.encode(
                [text], show_progress_bar=False, normalize_embeddings=True
            )[0]
            return emb
        except Exception as e:
            logger.warning("Failed to encode context for experience: %s", e)
            return None

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
                logger.info("ExperienceStore using shared bge-m3 (dim=1024)")
            else:
                logger.warning(
                    "Shared embedding model not available — "
                    "experience episodes will have no context embedding"
                )
        except ImportError:
            logger.warning(
                "model_registry not available — "
                "experience episodes will have no context embedding"
            )
            self._model = None
        except Exception as e:
            logger.warning("Failed to get shared embedding model for ExperienceStore: %s", e)
            self._model = None

        self._model_loaded = True

    # ═══════════════ PERSISTENCE ═══════════════

    def _record_threshold_change(self, old_threshold: float):
        """Record a threshold change in history for metrics computation.

        This is called internally whenever the adaptive threshold changes.
        The history enables threshold_health metric — is the threshold
        converging or oscillating?
        """
        self._threshold_history.append({
            'threshold': self._similarity_threshold,
            'timestamp': time.time(),
        })
        # Keep last 200 entries to prevent unbounded growth
        if len(self._threshold_history) > 200:
            self._threshold_history = self._threshold_history[-200:]

    def _save(self):
        """Persist episodes to disk with atomic write (v3 format)."""
        try:
            os.makedirs(os.path.dirname(self._store_path), exist_ok=True)

            data = {
                'version': 3,
                'episodes': self._episodes,
                'similarity_threshold': self._similarity_threshold,
                'match_history': self._match_history[-self.ADAPT_WINDOW_SIZE:],
                'threshold_history': self._threshold_history[-200:],
            }
            tmp_path = self._store_path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            try:
                os.replace(tmp_path, self._store_path)
            except OSError:
                try:
                    os.unlink(self._store_path)
                except OSError:
                    pass
                os.replace(tmp_path, self._store_path)
        except Exception as e:
            logger.warning("Failed to save experience store: %s", e)

    def _load(self) -> bool:
        """Load episodes from disk.

        Supports backward compatibility: v2 data is loaded with defaults
        for new fields (adjustment_applied=False, threshold_at_time=current).

        Returns True if episodes were loaded successfully.
        """
        if not os.path.exists(self._store_path):
            return False

        try:
            with open(self._store_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            version = data.get('version', 1)
            self._episodes = data.get('episodes', [])

            # Restore adaptive threshold state
            if data.get('similarity_threshold') is not None:
                self._similarity_threshold = max(
                    self.THRESHOLD_MIN,
                    min(self.THRESHOLD_MAX, data['similarity_threshold'])
                )
            if data.get('match_history'):
                self._match_history = data['match_history'][-self.ADAPT_WINDOW_SIZE:]
                # v2→v3 migration: add missing fields with defaults
                for entry in self._match_history:
                    entry.setdefault('adjustment_applied', False)
                    entry.setdefault('threshold_at_time', self._similarity_threshold)

            # v3 feature: threshold history
            if data.get('threshold_history'):
                self._threshold_history = data['threshold_history'][-200:]

            # Rebuild indices
            self._rebuild_indices()

            logger.info("Loaded %d experience episodes (v%d) from %s",
                        len(self._episodes), version, self._store_path)
            return len(self._episodes) > 0
        except Exception as e:
            logger.warning("Failed to load experience store: %s", e)
            return False

    # ═══════════════ INDEX MANAGEMENT ═══════════════

    def _rebuild_indices(self):
        """Rebuild node index and embedding cache from episodes list."""
        self._node_index = {}
        self._embedding_cache = {}

        for idx, episode in enumerate(self._episodes):
            node_id = episode.get('node_id', '')
            if node_id:
                if node_id not in self._node_index:
                    self._node_index[node_id] = []
                self._node_index[node_id].append(idx)

            # Rebuild embedding cache
            ep_id = episode.get('id', '')
            ep_emb_list = episode.get('context_embedding')
            if ep_emb_list is not None and ep_id:
                ep_emb = np.array(ep_emb_list, dtype=np.float32)
                norm = np.linalg.norm(ep_emb)
                if norm > 1e-8:
                    ep_emb = ep_emb / norm
                self._embedding_cache[ep_id] = ep_emb

    def _prune_oldest(self, count: int):
        """Remove the oldest `count` episodes (FIFO eviction)."""
        if count <= 0:
            return

        self._episodes = self._episodes[count:]

        # Rebuild all indices after removal
        self._rebuild_indices()


# ═══════════════ SINGLETON ═══════════════

_shared_store = None


def get_shared_store() -> ExperienceStore:
    """Get the shared ExperienceStore singleton.

    Prevents multiple instances from loading the model and data
    separately — saves RAM and ensures consistency.
    """
    global _shared_store
    if _shared_store is None:
        _shared_store = ExperienceStore()
    return _shared_store
