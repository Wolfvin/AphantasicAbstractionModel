# @WHO:   self-ai/src/derivation/text_comprehension.py
# @WHAT:  SELF-AI text comprehension — meta-learning, NOT hardcoded rules
# @PART:  self-ai/derivation
# @ENTRY: TextComprehension.comprehend(), TextComprehension.teach(), TextComprehension.teach_from_correction(), TextComprehension._self_evaluate_and_record(), TextComprehension.set_experience_enabled()

"""Text Comprehension — Meta-Learning AI yang memahami teks melalui pengamatan dan berpikir.

VISION:
    Meta-Learning AI yang fokus pada memahami dirinya sendiri, teks, dan
    membangun abstraction layernya sendiri berdasarkan pengamatan.

    Teacher provides:
        - Soal (problem/question)
        - Cara penyelesaian (solution steps)
        - Jawaban (answer)
        - Penjelasan kenapa (explanation of WHY)

    SELF discovers patterns through inner thinking, NOT from parsers.
    SELF builds its own concept clusters through teaching, not from hardcoded data.
    SELF classifies questions through SELF-discovered patterns and LLM reasoning.

    Architecture:
        - comprehend()   → main entry point
        - teach()        → teaching mechanism (SELF discovers, not hardcoded)
        - _has_concept() → concept detection (SELF-discovered concepts)
        - semantic_match() → embedding-based matching
        - PatternLearner → SELF discovers semantic patterns through inner thinking
        - TeachingLessons → structured teaching examples (soal + cara + jawaban + kenapa)
        - LLM fallback, counterfactual verification, meta-cognitive reflection
        - Persistence of learned patterns

    NO hardcoded concept clusters. SELF builds them through teaching.
    NO hardcoded question classifiers. SELF classifies via teaching + LLM.
    NO hardcoded linguistic rules. SELF discovers proposition patterns.
    NO hardcoded word lists, stop words, or semantic mappings.
"""

import os
import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from derivation.pattern_learner import PatternLearner
from derivation.teaching_lessons import TeachingLessons, TeachingLesson
from derivation.answer_handlers import AnswerHandlers


class TextComprehension:
    """Meta-Learning AI text comprehension — SELF discovers patterns, not hardcoded rules.

    SELF starts with empty concept clusters and builds them through teaching.
    SELF classifies questions through teaching lessons and LLM reasoning,
    not from hardcoded keyword matching.
    SELF extracts propositions through simple sentence splitting and
    SELF-discovered patterns, not from deleted parser modules.

    Key principles:
        1. NO hardcoded concept clusters — SELF builds them through teaching
        2. NO hardcoded question classification — SELF classifies via patterns + LLM
        3. NO hardcoded linguistic rules — SELF discovers proposition patterns
        4. ALL patterns are SELF-discovered through inner thinking
    """

    # ── System 2 Think Slow: Persistence & Strengthening ──
    # v38: Strengthening ONLY via external validation (provide_feedback).
    # v37: Learned patterns are PERMANENT — no temporal decay.
    # This is the "stack on top of parameters" — AI grows smarter each day.
    MAX_TEACHING_EXAMPLES = 500
    MAX_TEACHING_LESSONS = 500
    CONFIDENCE_BOOST_PER_VERIFY = 0.05   # +confidence when EXTERNAL feedback confirms correct
    CONFIDENCE_PENALTY_PER_FAILURE = 0.10  # -confidence when EXTERNAL feedback says wrong
    CONFIDENCE_CAP = 0.95               # Maximum confidence a pattern can reach
    CONFIDENCE_FLOOR = 0.1              # Below this, pattern is marked inactive
    CONFIDENCE_INACTIVE_THRESHOLD = 0.2 # Below this, pattern is skipped during matching

    def __init__(self, self_core=None, use_embeddings=True, use_llm=True):
        self.self_core = self_core
        self.propositions = []
        self.learned_patterns = {}
        self.teaching_examples = []

        # ── SELF-discovered concept clusters — starts EMPTY ──
        # SELF will build these through teaching, not from hardcoded data
        self._concept_clusters = {}

        # ── Teaching lessons — structured soal + cara + jawaban + kenapa ──
        self._lessons = TeachingLessons()

        # ── Pattern learner — SELF discovers patterns through inner thinking ──
        self._pattern_learner = None  # Lazy-init when LLM engine is available

        # ── Embedding-based concept detection ──
        self._embedding_detector = None
        self._use_embeddings = use_embeddings

        # ── LLM reasoning fallback ──
        self._use_llm = use_llm
        self._llm_engine = None

        # ── Experience store — episode-based learning for retrieval penalty ──
        # Qwen3 does NOT know about this. Only bge-m3 retrieval is affected.
        self._experience_store = None  # Lazy-init via ExperienceStore singleton

        # ── Ablation toggle — disable experience recording for testing ──
        # When disabled, _self_evaluate_and_record() becomes a no-op,
        # and record_feedback()/provide_feedback() skip experience recording.
        # This does NOT clear stored data — toggle back on to resume.
        self._experience_enabled = True

        # ── Prototype embeddings cache for question classification ──
        # Prevents re-encoding all lesson prototypes on every classify call.
        # Invalidated when new lessons are added (tracked by lesson count).
        self._proto_cache = None
        self._proto_lesson_count = 0

        # ── Persistence paths for learned knowledge ──
        self._patterns_file = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'learned_patterns.json'
        )

        # Load previously learned patterns
        self._load_learned_patterns()

        # ── Seed embedding patterns from default lessons ──
        # v41: Default lessons previously only fed UnderstandingComposer/Builder,
        # which creates understanding nodes but NOT embedding patterns. The
        # embedding matching system (_match_by_embedding) was blind at startup.
        # Now we call teach() for each default lesson so embedding patterns
        # are available from the start. Empirically verified: more training
        # data per subtype improves matching by +4-25% (ide_pokok +25%,
        # bahasa_kiasan +17%).
        self._seed_embedding_patterns()

        # ── Re-embed patterns that lack embeddings (model loaded after init) ──
        # v43: Patterns seeded without embedding model have empty context_embedding
        # and question_embedding. This makes _match_by_embedding() return None,
        # causing ALL queries to fall through to answer handlers (e.g.,
        # peribahasa_no_understanding). Re-embedding fixes this by populating
        # the missing embeddings once the model is available.
        self._reembed_patterns()

        # ── Answer handlers ──
        self._handlers = AnswerHandlers(self)

    # ═══════════════ PATTERN LEARNER INITIALIZATION ═══════════════

    def _get_pattern_learner(self):
        """Lazy-initialize the PatternLearner.

        PatternLearner needs the LLM engine for inner thinking.
        If LLM is not available, it falls back to basic observation.
        """
        if self._pattern_learner is None:
            try:
                # Try to get LLM engine first
                llm_engine = self._get_llm_engine()
                patterns_file = os.path.join(
                    os.path.dirname(__file__), '..', '..', 'data', 'self_discovered_patterns.json'
                )
                self._pattern_learner = PatternLearner(
                    llm_engine=llm_engine,
                    patterns_file=patterns_file,
                )
            except Exception as e:
                logger.warning("Failed to init PatternLearner: %s", e)
                # Create without LLM engine — will use basic observation
                patterns_file = os.path.join(
                    os.path.dirname(__file__), '..', '..', 'data', 'self_discovered_patterns.json'
                )
                self._pattern_learner = PatternLearner(patterns_file=patterns_file)
        return self._pattern_learner

    def _get_llm_engine(self):
        """Lazy-initialize the LLM reasoning engine."""
        if self._llm_engine is None and self._use_llm:
            try:
                from derivation.llm_reasoning import LLMReasoningEngine
                self._llm_engine = LLMReasoningEngine(confidence_threshold=0.5)
            except ImportError:
                logger.info("llm_reasoning module not available")
                self._use_llm = False
            except Exception as e:
                logger.warning("Failed to init LLMReasoningEngine: %s", e)
                self._use_llm = False
        return self._llm_engine

    # ═══════════════ DEFAULT LESSON SEEDING ═══════════════

    def _seed_embedding_patterns(self):
        """Seed embedding patterns from default lessons.

        v41: Default lessons previously only fed UnderstandingComposer/Builder,
        which creates understanding graph nodes but NOT embedding patterns.
        The embedding matching system (_match_by_embedding) was blind at
        startup — it only had patterns explicitly created via teach().

        This method calls teach() for each default lesson, ensuring
        embedding patterns are available from the start. Empirically:
          - Thin data (1-2 examples/subtype): 86% matching accuracy
          - Expanded data (5-6 examples/subtype): 90% matching accuracy
          - ide_pokok: +25%, bahasa_kiasan: +17%

        Only seeds patterns that don't already exist (avoids duplicates
        on repeated initialization). The embedding model is loaded lazily
        by teach() → _encode_teaching_semantic() — if unavailable,
        patterns get empty embeddings but text fields are still populated.
        """
        try:
            from derivation.teaching_lessons import get_default_lessons
            lessons = get_default_lessons()
        except Exception as e:
            logger.warning("Failed to load default lessons for seeding: %s", e)
            return

        if lessons.count() == 0:
            return

        # Build set of existing pattern keys to avoid duplicates
        existing_questions = set()
        for pk, pd in self.learned_patterns.items():
            q = pd.get('question_snippet', '')
            if q:
                existing_questions.add(q[:80])

        seeded = 0
        for lesson in lessons.get_all():
            # Skip if this lesson's question already has a pattern
            q_snippet = lesson.problem[:80]
            if q_snippet in existing_questions:
                continue

            # Call teach() with the lesson data — this creates the full
            # embedding pattern including context_embedding, question_embedding,
            # reasoning_embedding, etc. Pass question_type explicitly to avoid
            # misclassification by _classify_question() when models aren't loaded.
            try:
                self.teach(
                    text=lesson.context_text or lesson.problem,
                    question=lesson.problem,
                    correct_answer=lesson.answer,
                    explanation=lesson.explanation_why,
                    solution_steps=lesson.solution_steps,
                    question_type=lesson.question_type,
                )
                seeded += 1
            except Exception as e:
                logger.debug("Failed to seed lesson '%s': %s", q_snippet[:40], e)

        if seeded > 0:
            logger.info("Seeded %d embedding patterns from default lessons (total: %d)",
                        seeded, len(self.learned_patterns))

    def _reembed_patterns(self):
        """Re-embed patterns that lack embeddings — the primary PoC blocker fix.

        v43: When TextComprehension is initialized and the embedding model is
        NOT yet loaded (first startup, model downloading, etc.), _seed_embedding_patterns()
        creates patterns with empty context_embedding and question_embedding lists.
        This makes _match_by_embedding() return None for ALL queries, causing
        the system to always fall through to answer handlers that return
        "{type}_no_understanding" — learned patterns are on disk but invisible
        to the matching pipeline.

        This method fixes the problem by:
          1. Loading the embedding model (bge-m3)
          2. Iterating all learned patterns
          3. Re-encoding any pattern with empty embeddings
          4. Saving the updated patterns to disk

        Called once during __init__() after _seed_embedding_patterns().
        If the model is unavailable, silently skips (patterns will be
        re-embedded on next startup when the model IS available).
        """
        try:
            from derivation.model_registry import get_shared_embedding_model
            model = get_shared_embedding_model()
            if model is None:
                logger.info("No embedding model — skipping _reembed_patterns()")
                return
        except Exception as e:
            logger.info("Embedding model not available for re-embedding: %s", e)
            return

        reembedded = 0
        import numpy as np

        for pk, pd in self.learned_patterns.items():
            needs_context = not pd.get('context_embedding')
            needs_question = not pd.get('question_embedding')
            needs_reasoning = not pd.get('reasoning_embedding')
            needs_answer = not pd.get('answer_embedding')

            if not (needs_context or needs_question or needs_reasoning or needs_answer):
                continue  # This pattern already has embeddings

            # Re-encode the missing embeddings
            try:
                texts_to_encode = []
                text_keys = []

                if needs_question and pd.get('question_snippet'):
                    texts_to_encode.append(pd['question_snippet'])
                    text_keys.append('question_embedding')

                if needs_context and pd.get('text_snippet'):
                    texts_to_encode.append(pd['text_snippet'])
                    text_keys.append('context_embedding')

                if needs_reasoning and pd.get('reasoning_text'):
                    texts_to_encode.append(pd['reasoning_text'][:300])
                    text_keys.append('reasoning_embedding')

                if needs_answer and pd.get('answer'):
                    texts_to_encode.append(str(pd['answer']))
                    text_keys.append('answer_embedding')

                if not texts_to_encode:
                    continue

                embeddings = model.encode(
                    texts_to_encode, show_progress_bar=False,
                    normalize_embeddings=True
                )

                for i, key in enumerate(text_keys):
                    pd[key] = embeddings[i].tolist()

                # Also encode deductive pattern if missing
                if not pd.get('deductive_pattern_embedding') and pd.get('reasoning_text'):
                    dp_text = pd['reasoning_text'][:300]
                    dp_emb = model.encode(
                        [dp_text], show_progress_bar=False,
                        normalize_embeddings=True
                    )[0]
                    pd['deductive_pattern_embedding'] = dp_emb.tolist()
                    pd['deductive_pattern_text'] = dp_text

                reembedded += 1

            except Exception as e:
                logger.debug("Failed to re-embed pattern %s: %s", pk[:40], e)

        if reembedded > 0:
            self._save_learned_patterns()
            logger.info("v43: Re-embedded %d/%d patterns (model now available)",
                        reembedded, len(self.learned_patterns))

    # ═══════════════ CONCEPT METHODS ═══════════════

    def _has_concept(self, text: str, concept_path: str) -> bool:
        """Check if text contains a SELF-discovered concept — EMBEDDING ONLY.

        v38: Removed keyword matching path entirely. All concept detection
        is now done via cosine similarity in embedding space. This enforces
        the "no hardcore" principle — no keyword lists, no string matching,
        no regex patterns for concept detection.

        Strategy (embedding-only):
          1. Context embedding match — compare text against stored teaching
             context embeddings for this concept path (cosine sim >= 0.5)
          2. Embedding centroid match — compare text against concept centroid
             embeddings (cosine sim >= 0.45)
          3. Return True if ANY matches (OR logic)

        The keyword path (Step 1 in v37) was "hardcore in disguise" —
        splitting cluster names into words and doing regex matching
        is rule-based, not semantic. It violated the core principle.
        """
        # Step 1: Context embedding match (v35+: replaces ALL keyword matching)
        if self._match_context_embedding(text, concept_path, threshold=0.5):
            return True

        # Step 2: Embedding centroid match (semantic, catches paraphrases)
        detector = self._get_embedding_detector()
        if detector is not None:
            try:
                if detector.has_concept(text, concept_path, threshold=0.45):
                    return True
            except Exception:
                pass

        return False

    def _match_context_embedding(self, text: str, concept_path: str,
                                  threshold: float = 0.5) -> bool:
        """Match text against stored context embeddings for a concept path.

        v35: This replaces the old text_signals keyword matching.
        Instead of checking if generic words appear in the text,
        we check if the text is semantically similar to any stored
        teaching context for this concept path.

        This is more precise because cosine similarity >= 0.5 requires
        actual semantic relatedness, not just word overlap.
        """
        parts = concept_path.split('.', 1)
        if len(parts) != 2:
            return False

        cluster_name, sub_name = parts
        cluster = self._concept_clusters.get(cluster_name, {})

        # Check context_embeddings subcluster
        ctx_entries = cluster.get('context_embeddings', [])
        if not ctx_entries:
            return False

        try:
            from derivation.model_registry import get_shared_embedding_model
            model = get_shared_embedding_model()
            if model is None:
                return False

            import numpy as np
            text_emb = model.encode(
                [text[:200]], show_progress_bar=False,
                normalize_embeddings=True
            )[0]

            for entry in ctx_entries:
                emb_list = entry.get('embedding', [])
                if not emb_list:
                    continue
                ctx_emb = np.array(emb_list)
                # Normalize in case it wasn't stored normalized
                norm = np.linalg.norm(ctx_emb)
                if norm > 1e-8:
                    ctx_emb = ctx_emb / norm
                sim = float(np.dot(text_emb, ctx_emb))
                if sim >= threshold:
                    return True
        except Exception:
            pass

        return False

    def _extract_concept_words(self, text: str, concept_path: str) -> list:
        """Extract concept words from SELF-discovered concept clusters.

        v35: Skip context_embeddings subclusters (dict entries, not word lists).
        """
        parts = concept_path.split('.', 1)
        if len(parts) != 2:
            return []
        words_list = self._concept_clusters.get(parts[0], {}).get(parts[1], [])
        if not words_list:
            return []
        # Skip context_embeddings — they're dicts, not word lists
        if parts[1] == 'context_embeddings':
            return []
        text_lower = text.lower()
        result = []
        for w in words_list:
            if not isinstance(w, str):
                continue  # Skip non-string entries
            if len(w) >= 3:
                pattern = re.compile(r'\b' + re.escape(w) + r'\b')
                if pattern.search(text_lower):
                    result.append(w)
            elif w in text_lower:
                result.append(w)
        return result

    def _get_embedding_detector(self):
        """Lazy load embedding-based concept detector.

        Returns None if embeddings are disabled or unavailable,
        so callers can fall back to keyword matching.
        """
        if self._embedding_detector is None and self._use_embeddings:
            try:
                from derivation.embedding_concepts import EmbeddingConceptDetector
                cache_dir = os.path.join(
                    os.path.dirname(__file__), '..', '..', 'data', 'embedding_cache'
                )
                self._embedding_detector = EmbeddingConceptDetector(
                    concept_clusters=self._concept_clusters,
                    cache_dir=cache_dir,
                )
            except ImportError:
                logger.info("embedding_concepts not available — using keyword fallback")
                self._use_embeddings = False
            except Exception as e:
                logger.warning("Failed to init EmbeddingConceptDetector: %s — using keyword fallback", e)
                self._use_embeddings = False
        return self._embedding_detector

    def _detect_concepts(self, text: str) -> dict:
        """Detect SELF-discovered concepts in text — EMBEDDING ONLY.

        v38: Removed keyword detection path. All concept detection is
        now done via embedding similarity. This enforces "no hardcore" —
        no regex matching, no word overlap, no string comparison.

        Strategy:
          1. Context embedding match — compare text against all stored
             teaching context embeddings (cosine sim >= 0.4)
          2. Embedding centroid match — compare text against concept centroids
          3. Combine results from both paths
        """
        detected = {}

        # Step 1: Context embedding matching across all clusters
        try:
            from derivation.model_registry import get_shared_embedding_model
            import numpy as np
            model = get_shared_embedding_model()
            if model is not None:
                text_emb = model.encode(
                    [text[:300]], show_progress_bar=False,
                    normalize_embeddings=True
                )[0]

                for cn, subs in self._concept_clusters.items():
                    ctx_entries = subs.get('context_embeddings', [])
                    if not ctx_entries:
                        continue

                    best_sim = 0.0
                    best_answer = ''
                    for entry in ctx_entries:
                        emb_list = entry.get('embedding', [])
                        if not emb_list:
                            continue
                        ctx_emb = np.array(emb_list)
                        norm = np.linalg.norm(ctx_emb)
                        if norm > 1e-8:
                            ctx_emb = ctx_emb / norm
                        sim = float(np.dot(text_emb, ctx_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_answer = entry.get('answer', '')

                    if best_sim >= 0.4:
                        # Match all sub-paths under this cluster
                        for sn in subs:
                            if sn == 'context_embeddings':
                                continue
                            path = f"{cn}.{sn}"
                            detected[path] = [f'emb_match:{best_sim:.2f}']
        except Exception:
            pass

        # Step 2: Embedding centroid detection (catches paraphrases)
        detector = self._get_embedding_detector()
        if detector is not None:
            try:
                emb_result = detector.detect_concepts(text, top_k=10, threshold=0.35)
                if emb_result:
                    for path, words in emb_result.items():
                        if path not in detected:
                            detected[path] = words
            except Exception:
                pass

        return detected

    def _detect_concepts_for_matching(self, text: str, question: str) -> dict:
        """Detect concepts for the purpose of narrowing pattern candidates.

        v39: Combines context text and question for concept detection,
        ensuring both context-level markers (e.g., "menari-nari" in text)
        and question-level markers (e.g., "kata menari-nari" in question)
        are detected. This feeds into _concept_narrowed_pool() to prevent
        cross-subtype false positives during embedding matching.

        Returns:
            Dict of {concept_path: [matched_words]} — same format as
            _detect_concepts(). Empty dict if detection fails.
        """
        # Detect in context (primary — where linguistic markers live)
        context_concepts = self._detect_concepts(text[:500])

        # Detect in question (secondary — may contain the target word)
        question_concepts = self._detect_concepts(question)

        # Merge: context concepts take priority, question fills gaps
        merged = dict(context_concepts)
        for path, words in question_concepts.items():
            if path not in merged:
                merged[path] = words
            else:
                # Add any new words from question detection
                existing = set(merged[path])
                for w in words:
                    if w not in existing:
                        merged[path].append(w)

        return merged

    def _concept_narrowed_pool(self, detected_concepts: dict,
                                q_type: str = None) -> dict:
        """Filter candidate patterns by concept cluster overlap.

        v39: Pre-filters learned patterns based on whether the input's
        detected concept paths overlap with each pattern's required_concepts.
        This prevents cross-subtype false positives — e.g., a simile query
        ("bagai rembulan") won't match a personifikasi pattern because
        their concept paths (figurative_language.simile vs
        figurative_language.personification_verb) don't overlap.

        Falls back to all patterns if:
          - No concepts detected in input (can't narrow)
          - No patterns pass the filter (incomplete concept coverage)

        Args:
            detected_concepts: Dict {concept_path: [words]} from
                _detect_concepts_for_matching()
            q_type: If set, also filter by question_type.
                None = cross-type allowed (for Pass 2/3).

        Returns:
            Dict of {pattern_key: pattern_data} for matching candidates.
        """
        # Build base pool filtered by q_type and active status
        base_pool = {}
        for pk, pd in self.learned_patterns.items():
            if pd.get('active', True) is False:
                continue
            if q_type is not None and pd.get('question_type') != q_type:
                continue
            base_pool[pk] = pd

        if not detected_concepts:
            # No concepts detected — can't narrow, use base pool
            return base_pool

        detected_paths = set(detected_concepts.keys())
        candidates = {}
        unfiltered = {}  # Patterns without concept info

        for pk, pd in base_pool.items():
            required = set(pd.get('required_concepts', []))
            generalized = set(pd.get('generalized_concepts', []))
            all_pattern_concepts = required | generalized

            if not all_pattern_concepts:
                # Pattern has no concept info — keep separately
                unfiltered[pk] = pd
                continue

            # Check if ANY detected concept path overlaps with pattern's concepts
            overlap = detected_paths & all_pattern_concepts
            if overlap:
                candidates[pk] = pd

        # Always include patterns without concept info (can't filter them)
        candidates.update(unfiltered)

        # Graceful fallback: if concept filter removed ALL candidates,
        # return base pool (don't miss matches due to incomplete concept
        # coverage — the data is still being built through teaching)
        if not candidates:
            logger.debug(
                "v39 concept filter excluded all %d patterns for q_type=%s — "
                "falling back to full pool",
                len(base_pool), q_type
            )
            return base_pool

        logger.debug(
            "v39 concept filter: %d/%d candidates for q_type=%s "
            "(detected: %s)",
            len(candidates), len(base_pool), q_type,
            list(detected_paths)[:5]
        )
        return candidates

    def semantic_match(self, answer: str, expected_meanings: list, threshold: float = 0.6) -> dict:
        """Match an answer against expected meanings using EMBEDDING similarity.

        Instead of checking if a keyword string appears in the answer,
        we encode both the answer and each expected meaning and compare
        cosine similarity. This is vector-based answer validation.

        Args:
            answer: The system's answer string
            expected_meanings: List of acceptable meaning strings
            threshold: Minimum cosine similarity to consider a match (0.0-1.0)

        Returns:
            dict with 'matched' (bool), 'best_match' (str), 'similarity' (float)
        """
        if not answer or not expected_meanings:
            return {'matched': False, 'best_match': None, 'similarity': 0.0}

        answer_lower = answer.lower().strip()

        # Step 1: Quick keyword check first (fast path)
        for meaning in expected_meanings:
            meaning_lower = meaning.lower().strip()
            if meaning_lower in answer_lower or answer_lower in meaning_lower:
                return {'matched': True, 'best_match': meaning, 'similarity': 1.0,
                        'match_type': 'keyword'}

        # Step 2: Embedding similarity check (semantic path)
        detector = self._get_embedding_detector()
        if detector is None or detector.model is None:
            return {'matched': False, 'best_match': None, 'similarity': 0.0,
                    'match_type': 'keyword_fallback'}

        try:
            all_texts = [answer_lower] + [m.lower().strip() for m in expected_meanings]
            embeddings = detector.model.encode(all_texts, show_progress_bar=False,
                                                normalize_embeddings=True)
            answer_emb = embeddings[0]
            meaning_embs = embeddings[1:]

            import numpy as np
            similarities = [float(np.dot(answer_emb, mb)) for mb in meaning_embs]
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]

            return {
                'matched': best_sim >= threshold,
                'best_match': expected_meanings[best_idx],
                'similarity': best_sim,
                'match_type': 'embedding',
                'all_similarities': dict(zip(expected_meanings, similarities)),
            }
        except Exception as e:
            logger.warning("Semantic match error: %s", e)
            return {'matched': False, 'best_match': None, 'similarity': 0.0,
                    'match_type': 'error'}

    def _semantic_validate_answer(self, answer: dict, text: str, question: str) -> dict:
        """Validate answer relevance using embedding similarity.

        Checks if the answer is semantically related to the question.
        If the answer seems unrelated, reduce confidence.
        """
        answer_text = answer.get('answer', '')
        if not answer_text or not isinstance(answer_text, str):
            return answer

        ql = question.lower()
        al = answer_text.lower()

        # For short answers, validate with embedding similarity
        answer_words = al.split()
        if len(answer_words) <= 2:
            detector = self._get_embedding_detector()
            if detector is None or detector.model is None:
                return answer

            try:
                import numpy as np
                embeddings = detector.model.encode(
                    [ql, al], show_progress_bar=False, normalize_embeddings=True
                )
                q_emb, a_emb = embeddings[0], embeddings[1]
                similarity = float(np.dot(q_emb, a_emb))

                if similarity < 0.3:
                    answer['confidence'] = answer.get('confidence', 0.5) * 0.5
                    answer['semantic_validation'] = f'low_relevance:{similarity:.2f}'
                elif similarity < 0.5:
                    answer['confidence'] = answer.get('confidence', 0.5) * 0.85
                    answer['semantic_validation'] = f'moderate_relevance:{similarity:.2f}'
                else:
                    answer['semantic_validation'] = f'high_relevance:{similarity:.2f}'
            except Exception:
                pass

        return answer

    def _match_learned_pattern(self, text: str, question: str, q_type: str) -> Optional[dict]:
        """Match learned patterns — EMBEDDING-FIRST, then concept fallback.

        v36: Semantic Teaching Protocol — matching is now embedding-first.
        The primary matching strategy uses cosine similarity between the
        question's embedding and stored question embeddings. This is fully
        semantic — no keyword or text matching involved.

        Strategy 0 (PRIMARY): Embedding similarity matching (v36)
          - Encode the question with bge-m3
          - Compare against all stored question_embeddings (cosine similarity)
          - High similarity (> 0.85) → EXACT match → return stored answer directly
          - Medium similarity (0.5–0.85) → GENERALIZATION → derive via LLM with reasoning context
          - Low similarity (< 0.5) → fall through to concept matching

        Strategy 1 (FALLBACK): Required concepts match (from SELF-built clusters)
        Strategy 2 (FALLBACK): Generalized concepts match (broader)
        Strategy 3 (LEGACY): SELF-discovered pattern via PatternLearner

        Qwen3 never sees the teaching data directly. For generalization,
        the reasoning_text is passed as context, but Qwen3 doesn't know
        it came from a teaching — it's just additional reasoning context.
        """
        # ── Strategy 0 (v36): Embedding-first semantic matching ──
        emb_match = self._match_by_embedding(text, question, q_type)
        if emb_match is not None:
            return emb_match

        # ── Strategy 1 (fallback): Required concepts match ──
        candidates = []

        for pk, pd in self.learned_patterns.items():
            # v38: Skip inactive patterns (confidence too low from repeated failures)
            if pd.get('active', True) is False:
                continue

            if pd.get('question_type') != q_type:
                continue

            # Required concepts match
            required = pd.get('required_concepts', [])
            if required:
                matched = sum(1 for rc in required if self._has_concept(text, rc))
                threshold = max(1, len(required) * 0.3)
                if matched >= threshold:
                    match_ratio = matched / len(required) if required else 0
                    specificity = match_ratio * 2.0  # Required concepts weighted 2x
                    confidence = pd.get('confidence', 0.55) + 0.05 * matched
                    candidates.append({
                        'answer': pd.get('answer_template', pd.get('answer', '')),
                        'confidence': confidence,
                        'method': f'{q_type}_learned',
                        'explanation': f"Learned pattern matched ({matched}/{len(required)} concepts): {pk}",
                        'specificity': specificity,
                    })

            # Generalized concepts match
            generalized = pd.get('generalized_concepts', [])
            if generalized:
                gen_matched = sum(1 for gc in generalized if self._has_concept(text, gc))
                gen_threshold = max(3, len(generalized) * 0.3)
                if gen_matched >= gen_threshold:
                    match_ratio = gen_matched / len(generalized) if generalized else 0
                    specificity = match_ratio * 1.0  # Generalized concepts weighted 1x
                    confidence = pd.get('confidence', 0.45) + 0.03 * gen_matched
                    candidates.append({
                        'answer': pd.get('answer_template', pd.get('answer', '')),
                        'confidence': confidence,
                        'method': f'{q_type}_learned_generalized',
                        'explanation': f"Generalized pattern matched ({gen_matched}/{len(generalized)} concepts): {pk}",
                        'specificity': specificity,
                    })

        # Return the most specific match
        if candidates:
            best = max(candidates, key=lambda c: c['specificity'])
            best.pop('specificity', None)
            return best

        # Strategy 3 (legacy): Try PatternLearner
        pattern_learner = self._get_pattern_learner()
        if pattern_learner is not None:
            matching_pattern = pattern_learner.find_matching_pattern(text, question, q_type)
            if matching_pattern is not None:
                result = pattern_learner.apply_pattern(matching_pattern, text, question, q_type)
                if result is not None:
                    return result

        return None

    def _match_by_embedding(self, text: str, question: str, q_type: str) -> Optional[dict]:
        """Match question against stored embeddings via cosine similarity.

        v40b: Variance-weighted signal combination. No single signal is
        optimal for all domains — c→c is discriminative for majas (where
        linguistic markers live in context), while q→q is discriminative
        for ide_pokok/penyebab_akibat (where the question determines the
        answer type). Instead of hardcoding which signal to use, we let
        the data decide per query:

        1. Compute c→c and q→q similarity scores against all candidates
        2. Measure variance of each score distribution
        3. Weight each signal by its variance (higher variance = more
           discriminative = higher weight)
        4. Combined score = w_c * c→c + w_q * q→q

        Empirical results (20 test cases, 5 domains):
          - v38 max(q→q, c→c, c→r): 16/20 (80%)
          - v40 c→c only:            13/20 (65%) — regression on ide_pokok
          - v40b variance-weighted:  16/20 (80%) — no regression

        Matching passes:
          Pass 1: Variance-weighted combined (c→c + q→q)
            - combined >= 0.5: match found → derive via LLM

          Pass 2: Question → Question embedding (EXACT only)
            - cosine > 0.85: same question repeated → return stored answer

          Pass 3: Context → reasoning/deductive pattern embeddings
            - cosine > 0.4: reasoning pattern match → derive via LLM

        Qwen3 doesn't know it was "taught" — it just receives richer
        reasoning context that helps it derive the correct answer.
        """
        try:
            from derivation.model_registry import get_shared_embedding_model
            import numpy as np

            model = get_shared_embedding_model()
            if model is None:
                return None

            # Encode the incoming question and context
            texts_to_encode = [question, text[:300]]
            encoded = model.encode(texts_to_encode, show_progress_bar=False,
                                   normalize_embeddings=True)
            q_emb = encoded[0]
            ctx_emb = encoded[1]

            # Build candidate pool (active, same q_type)
            candidate_pool = {
                pk: pd for pk, pd in self.learned_patterns.items()
                if pd.get('active', True) is not False
                and pd.get('question_type') == q_type
            }

            if not candidate_pool:
                return None

            # ── Pass 1: Variance-weighted combined matching ──
            # Compute c→c and q→q scores for all candidates, then weight
            # by variance — the signal with more spread is more discriminative.
            c_scores = {}
            q_scores = {}

            for pk, pd in candidate_pool.items():
                # Context-to-context score
                ctx_emb_stored = pd.get('context_embedding', [])
                if ctx_emb_stored:
                    stored = np.array(ctx_emb_stored)
                    norm = np.linalg.norm(stored)
                    if norm > 1e-8:
                        stored = stored / norm
                    c_scores[pk] = float(np.dot(ctx_emb, stored))

                # Question-to-question score
                q_emb_stored = pd.get('question_embedding', [])
                if q_emb_stored:
                    stored = np.array(q_emb_stored)
                    norm = np.linalg.norm(stored)
                    if norm > 1e-8:
                        stored = stored / norm
                    q_scores[pk] = float(np.dot(q_emb, stored))

            if c_scores or q_scores:
                # Compute variance of each score distribution
                c_var = float(np.var(list(c_scores.values()))) if len(c_scores) > 1 else 0.0
                q_var = float(np.var(list(q_scores.values()))) if len(q_scores) > 1 else 0.0

                total_var = c_var + q_var
                if total_var > 1e-8:
                    w_c = c_var / total_var
                    w_q = q_var / total_var
                else:
                    # No variance — equal weight fallback
                    w_c = 0.5
                    w_q = 0.5

                # Compute combined score for each candidate
                combined_scores = {}
                for pk in candidate_pool:
                    c = c_scores.get(pk, 0.0)
                    q = q_scores.get(pk, 0.0)
                    # If a pattern lacks one signal, use what's available
                    if pk not in c_scores:
                        combined_scores[pk] = q  # Only q→q available
                    elif pk not in q_scores:
                        combined_scores[pk] = c  # Only c→c available
                    else:
                        combined_scores[pk] = w_c * c + w_q * q

                # Pick the best combined match
                best_pk = max(combined_scores, key=combined_scores.get)
                best_combined = combined_scores[best_pk]
                best_pd = candidate_pool[best_pk]

                if best_combined >= 0.5:
                    # v43: Before LLM derivation, check q→q exact match.
                    # If q→q similarity is high (> 0.85), return the stored
                    # answer directly — it's the same question, no need for
                    # LLM generalization (which can produce garbage with
                    # small models like Qwen3-0.6B).
                    best_q_in_pass1 = q_scores.get(best_pk, 0.0)
                    if best_q_in_pass1 > 0.85:
                        confidence = min(0.95, best_q_in_pass1) * best_pd.get('confidence', 0.6)
                        return {
                            'answer': best_pd.get('answer', best_pd.get('answer_template', '')),
                            'confidence': confidence,
                            'method': f'{q_type}_learned_semantic_exact',
                            'explanation': f"Semantic exact match in Pass 1 (q→q={best_q_in_pass1:.3f}, combined={best_combined:.3f}): {best_pk}",
                            'semantic_similarity': best_q_in_pass1,
                        }

                    derived = self._derive_from_reasoning(
                        text, question, q_type, best_pd, best_combined
                    )
                    if derived is not None:
                        return derived

            # ── Pass 2: Question → Question embedding (EXACT only) ──
            # For verbatim question repeats (sim > 0.85), return answer directly.
            best_q_pk = None
            best_q_sim = 0.0
            best_q_pd = None

            for pk, pd in candidate_pool.items():
                q_emb_stored = pd.get('question_embedding', [])
                if not q_emb_stored:
                    continue

                stored = np.array(q_emb_stored)
                norm = np.linalg.norm(stored)
                if norm > 1e-8:
                    stored = stored / norm
                sim = float(np.dot(q_emb, stored))

                if sim > best_q_sim:
                    best_q_sim = sim
                    best_q_pk = pk
                    best_q_pd = pd

            if best_q_pd is not None and best_q_sim > 0.85:
                confidence = min(0.95, best_q_sim) * best_q_pd.get('confidence', 0.6)
                return {
                    'answer': best_q_pd.get('answer', best_q_pd.get('answer_template', '')),
                    'confidence': confidence,
                    'method': f'{q_type}_learned_semantic_exact',
                    'explanation': f"Semantic exact match (sim={best_q_sim:.3f}): {best_q_pk}",
                    'semantic_similarity': best_q_sim,
                }

            # ── Pass 3: Reasoning/deductive pattern matching (cross-domain) ──
            # Broader pool: any q_type, but still active.
            broad_pool = {
                pk: pd for pk, pd in self.learned_patterns.items()
                if pd.get('active', True) is not False
            }

            best_reasoning_sim = 0.0
            best_reasoning_pd = None

            for pk, pd in broad_pool.items():
                r_emb_stored = pd.get('reasoning_embedding', [])
                dp_emb_stored = pd.get('deductive_pattern_embedding', [])

                best_sub_sim = 0.0
                for emb_list in [r_emb_stored, dp_emb_stored]:
                    if not emb_list:
                        continue
                    stored = np.array(emb_list)
                    norm = np.linalg.norm(stored)
                    if norm > 1e-8:
                        stored = stored / norm
                    sim = float(np.dot(ctx_emb, stored))
                    if sim > best_sub_sim:
                        best_sub_sim = sim

                if best_sub_sim > best_reasoning_sim:
                    best_reasoning_sim = best_sub_sim
                    best_reasoning_pd = pd

            if best_reasoning_pd is not None and best_reasoning_sim >= 0.4:
                derived = self._derive_from_reasoning(
                    text, question, q_type, best_reasoning_pd, best_reasoning_sim
                )
                if derived is not None:
                    return derived

            # ── No embedding match found — fall through to concept matching ──
            return None

        except Exception as e:
            logger.warning("Embedding matching failed: %s — falling back to concept matching", e)
            return None

    def _derive_from_reasoning(self, text: str, question: str,
                                q_type: str, matched_pattern: dict,
                                similarity: float) -> Optional[dict]:
        """Derive answer for a SIMILAR question using the LLM + reasoning context.

        v36: When a question is semantically similar to a taught question
        (0.5 <= cosine < 0.85), we use the LLM to derive the answer
        using the stored reasoning as context.

        KEY INSIGHT: For generalization, we must ABSTRACT the reasoning pattern
        before passing it to the LLM. If we pass the raw reasoning (which
        contains specific numbers like "3 kelereng"), the LLM will just
        repeat those numbers instead of applying the PATTERN to the new question.

        So we:
          1. Extract the ABSTRACT reasoning pattern (remove specific entities/numbers)
          2. Pass the abstract pattern + current question to the LLM
          3. The LLM applies the abstract pattern to derive the answer
          4. Validate the derived answer against the answer_embedding

        Qwen3 doesn't know it was "taught" — it just receives an abstract
        reasoning framework that helps it solve the current question.
        """
        reasoning_text = matched_pattern.get('reasoning_text', '')
        key_points = matched_pattern.get('key_points_text', [])
        deductive_pattern = matched_pattern.get('deductive_pattern_text', '')

        if not reasoning_text and not deductive_pattern:
            return None

        # ── Abstract the reasoning pattern ──
        # Instead of removing specific values (which makes the LLM output "N"),
        # we keep the full reasoning but frame it as a METHOD from a DIFFERENT example.
        # The LLM should apply the SAME METHOD to the current question.
        # Key: explicitly tell the LLM the example is from a DIFFERENT question.
        reasoning_context = ""
        if deductive_pattern:
            reasoning_context += f"Similar problem's reasoning: {deductive_pattern}\n"
        if key_points:
            reasoning_context += f"Similar problem's steps: {' → '.join(key_points)}\n"
        if reasoning_text:
            reasoning_context += f"Similar problem's explanation: {reasoning_text}\n"

        # Use LLM to derive answer with reasoning context
        try:
            llm_engine = self._get_llm_engine()
            if llm_engine is None:
                return None

            # Create a prompt that frames the reasoning as a METHOD from a different example
            # The LLM should apply the SAME reasoning METHOD to the current question,
            # not copy the specific values from the example.
            prompt = (
                f"Text: {text}\n\n"
                f"Question: {question}\n\n"
                f"A similar type of problem was solved before with this approach:\n"
                f"{reasoning_context}\n"
                f"Use the SAME REASONING METHOD to solve the current question. "
                f"Answer based on the TEXT above, not the example.\n"
                f"Answer:"
            )

            # Try local Qwen directly — bypass _try_local_qwen() because
            # its _parse_response() may strip the actual answer.
            # We need the RAW response for reliable number extraction.
            answer_text = None
            try:
                from derivation.model_registry import get_shared_qwen
                qwen_model, qwen_tokenizer = get_shared_qwen()
                if qwen_model is not None and qwen_tokenizer is not None:
                    inputs = qwen_tokenizer(
                        prompt, return_tensors='pt', max_length=512, truncation=True
                    )
                    # Move to model device
                    inputs = {k: v.to(qwen_model.device) for k, v in inputs.items()}
                    outputs = qwen_model.generate(
                        **inputs, max_new_tokens=100, temperature=0.3
                    )
                    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    answer_text = qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)
            except Exception as e:
                logger.warning("Direct Qwen call in _derive_from_reasoning failed: %s", e)

            # Fallback to _try_local_qwen if direct call failed
            if not answer_text:
                if hasattr(llm_engine, '_try_local_qwen'):
                    answer_text = llm_engine._try_local_qwen(prompt)

            if not answer_text:
                return None

            # Clean up the answer
            answer_clean = answer_text.strip()
            # Extract just the first line/number if it's verbose
            first_line = answer_clean.split('\n')[0].strip()

            # Try to extract a number from the answer
            import re as _re
            num_match = _re.search(r'\d+', first_line)
            if num_match:
                first_line = num_match.group(0)

            # Validate the answer against the stored answer_embedding
            # (semantic similarity check — is the derived answer similar to the taught answer?)
            answer_sim = 0.0
            answer_emb_stored = matched_pattern.get('answer_embedding', [])
            if answer_emb_stored:
                try:
                    from derivation.model_registry import get_shared_embedding_model
                    import numpy as np
                    model = get_shared_embedding_model()
                    if model is not None:
                        a_emb = model.encode([first_line], show_progress_bar=False,
                                             normalize_embeddings=True)[0]
                        stored = np.array(answer_emb_stored)
                        norm = np.linalg.norm(stored)
                        if norm > 1e-8:
                            stored = stored / norm
                        answer_sim = float(np.dot(a_emb, stored))
                except Exception:
                    pass

            # For generalization, the answer SHOULD be different from the taught answer,
            # so we DON'T reject based on low answer_sim. We just note it.
            # Confidence formula: base 0.4 + similarity contribution + answer validation bonus
            # The base is high enough to prevent LLM fallback override (needs > 0.5)
            confidence = 0.4 + similarity * 0.4 + min(answer_sim, 0.2)
            confidence = min(0.85, max(0.4, confidence))

            return {
                'answer': first_line,
                'confidence': confidence,
                'method': f'{q_type}_learned_semantic_generalized',
                'explanation': f"Semantic generalization (sim={similarity:.3f}, answer_sim={answer_sim:.3f})",
                'semantic_similarity': similarity,
                'answer_semantic_similarity': answer_sim,
            }

        except Exception as e:
            logger.warning("LLM derivation from reasoning failed: %s", e)
            return None

    def _abstract_reasoning(self, text: str) -> str:
        """Abstract specific values from reasoning text for generalization.

        v36: When generalizing, we must remove specific numbers, names,
        and entities so the LLM applies the PATTERN, not the specific values.

        Examples:
          "3 kelereng tidak berwarna merah" → "X tidak berwarna Y"
          "10 kelereng" → "N total items"
          "Andi" → "[subject]"

        This is a SEMANTIC abstraction — we don't use hardcoded word lists.
        Instead, we use regex patterns for numbers and common entity patterns.
        """
        if not text:
            return text

        import re as _re

        result = text

        # Replace specific numbers with abstract placeholders
        # But keep structural words like "kecuali", "tidak", "semua"
        result = _re.sub(r'\b\d+\b', 'N', result)

        # Replace quoted specific values with abstract placeholders
        result = _re.sub(r"'[^']+'", "'X'", result)

        return result

    # ═══════════════ MAIN ENTRY ═══════════════

    def comprehend(self, text: str, question: str) -> dict:
        """Main entry point — comprehend text and answer a question.

        Flow:
          1. Extract propositions (simple sentence splitting, no parser)
          2. Classify question (SELF-discovered patterns + LLM)
          3. Find answer via handlers, learned patterns, or LLM
          4. Meta-cognitive reflection if confidence is low
          5. Counterfactual verification
          6. Semantic validation
          7. Confidence calibration
        """
        propositions = self._extract_propositions(text)
        q_type = self._classify_question(question, text)
        answer = self._find_answer(propositions, question, q_type, text)

        # Meta-Cognitive Reflection — identify WHAT knowledge is missing
        if answer.get('confidence', 0) < 0.5 or answer.get('answer') is None:
            self._reflect_on_failure(text, question, answer, q_type, propositions)

        # Targeted LLM fallback — use meta-cognitive gap identification
        if self._use_llm and self._should_use_llm(answer):
            llm_answer = self._try_targeted_llm_reasoning(text, question, answer, q_type)
            if llm_answer.get('answer') is not None:
                if answer.get('answer') is None or answer.get('confidence', 0) < 0.4:
                    answer = llm_answer
                elif llm_answer.get('confidence', 0) > answer.get('confidence', 0):
                    answer = llm_answer
        # v31 fix (P3): Removed dead else branch — the condition inside was
        # IDENTICAL to the if condition, so it was never reachable. If targeted
        # LLM reasoning fails, the generic _try_llm_reasoning is already called
        # inside _try_targeted_llm_reasoning as its own fallback.

        # Counterfactual Verification — proof by contradiction
        if answer.get('answer') is not None:
            answer = self._verify_via_counterfactual(answer, text, propositions, q_type)

        # Semantic answer validation — verify answer relevance using embeddings
        if answer.get('answer') is not None and self._use_embeddings:
            answer = self._semantic_validate_answer(answer, text, question)

        # Calibrate confidence using Platt Scaling
        raw_confidence = answer.get('confidence', 0.0)
        calibrated = self._calibrate_confidence(raw_confidence)

        result = {'answer': answer.get('answer'), 'confidence': calibrated,
                  'raw_confidence': raw_confidence,
                  'method': answer.get('method', 'unknown'), 'explanation': answer.get('explanation', ''),
                  'question_type': q_type, 'propositions': propositions,
                  'counterfactual_verified': answer.get('counterfactual_verified', False)}

        # ── Self-evaluation: automatic feedback loop ──
        # SELF evaluates its own answer quality and records suspected failures
        # WITHOUT needing external input. This is the automatic feedback loop
        # that makes ExperienceWeight truly self-sustaining.
        # Qwen3 does NOT know about this — only bge-m3 retrieval is affected.
        if result.get('answer') is not None:
            self._self_evaluate_and_record(text, question, result, q_type)

        # v38: REMOVED _strengthen_pattern_if_verified() from here.
        # Strengthening based on confidence ≠ correctness. It created a
        # positive feedback loop where wrong-but-confident patterns got stronger.
        # Strengthening now ONLY happens via provide_feedback() when external
        # validation confirms the answer is correct.

        return result

    # ═══════════════ PROPOSITION EXTRACTION ═══════════════

    def _extract_propositions(self, text: str) -> list:
        """Extract propositions using simple sentence splitting.

        SELF discovers patterns through inner thinking, NOT from parsers.
        Since grammar/parser.py was deleted, we use simple sentence
        splitting and basic pattern matching.

        No hardcoded linguistic rules — SELF discovers proposition patterns
        through observation and teaching.
        """
        sentences = self._split_sentences(text)
        propositions = []

        for i, sentence in enumerate(sentences):
            prop = {
                'raw': sentence,
                'position': i,
                'text_lower': sentence.lower(),
            }

            propositions.append(prop)

        return propositions

    # ═══════════════ QUESTION CLASSIFICATION ═══════════════

    def _classify_question(self, question: str, text: str = '') -> str:
        """Classify question type using SELF-discovered patterns and LLM reasoning.

        Instead of hardcoded keyword matching from a deleted QuestionClassifier,
        this method uses:
          1. SELF-discovered patterns from teaching lessons
          2. LLM reasoning when available
          3. Simple signal-based classification as fallback

        NO hardcoded keyword lists. SELF discovers classification patterns
        through teaching.
        """
        question_lower = question.lower()

        # Step 1: Try SELF-discovered pattern matching for classification
        pattern_learner = self._get_pattern_learner()
        if pattern_learner is not None:
            # Use pattern-based understanding matching for classification
            for q_type in self._lessons.get_types():
                matching = pattern_learner.find_matching_pattern(text, question, q_type)
                if matching is not None:
                    return q_type

        # Step 2: Try LLM-based classification
        if self._use_llm:
            llm_result = self._classify_via_llm(question, text)
            if llm_result:
                return llm_result

        # Step 3: Simple signal-based classification (fallback)
        # These are minimal observations, NOT hardcoded keyword lists.
        # SELF will replace these with its own discovered patterns over time.
        return self._classify_by_signals(question_lower)

    def _classify_via_llm(self, question: str, text: str) -> Optional[str]:
        """Classify question type using LLM reasoning.

        SELF uses its inner thinking (LLM) to determine the question type
        based on the teaching examples it has observed.
        """
        llm_engine = self._get_llm_engine()
        if llm_engine is None:
            return None

        # Build a prompt with SELF's teaching context
        known_types = self._lessons.get_types()
        type_list = ', '.join(known_types) if known_types else (
            'eksplisit, implisit, interpretatif, ide_pokok, peribahasa, '
            'bahasa_kiasan, teks_argumentatif, perbandingan, motivasi, '
            'sinonim_antonim, sikap_tokoh, teks_prosedur, teks_persuasif, '
            'unsur_cerita, benar_salah, analogi, kesan_pesan, penyebab_ganda, '
            'pertanyaan_negatif, tone_mood, teks_eksplanasi, inferensi_silang, '
            'konteks_makna'
        )

        prompt = (
            f"Classify this question into one of these types: {type_list}\n"
            f"Question: {question}\n"
            f"Context text (if any): {text[:200]}\n"
            "Respond with ONLY the type name, nothing else."
        )

        try:
            answer = None
            if hasattr(llm_engine, '_try_sdk_fallback'):
                answer = llm_engine._try_sdk_fallback(text, question, prompt)
            if not answer and hasattr(llm_engine, '_try_local_qwen'):
                answer = llm_engine._try_local_qwen(prompt)

            if answer:
                # Clean and validate the answer
                answer = answer.strip().lower().replace(' ', '_').replace('-', '_')
                # Check if it's a valid type
                if known_types and answer in known_types:
                    return answer
                # Even if not in known types, return if it looks like a valid type
                if answer and len(answer) > 2 and answer.isidentifier():
                    return answer
        except Exception as e:
            logger.warning("LLM question classification error: %s", e)

        return None

    def _classify_by_signals(self, question_lower: str) -> str:
        """Embedding-based question classification as fallback.

        v32: Removed MASSIVE hardcoded signal→type map (40+ entries) that
        violated the "NO hardcoded keyword lists" philosophy.

        v35: Uses cached prototype embeddings to avoid re-encoding every call.
        Prototype embeddings are computed once when teaching lessons exist,
        then reused for all subsequent classifications. This prevents the
        expensive per-call re-encoding that caused timeouts.

        Strategy:
          1. Embedding similarity against cached lesson prototypes (fast)
          2. LLM-based classification (slower but accurate)
          3. Default to 'eksplisit' only as last resort

        SELF will learn better classification through the self-correction
        loop when it misclassifies questions.
        """
        # Step 1: Try embedding-based classification with cached prototypes
        try:
            detector = self._get_embedding_detector()
            if detector is not None and detector.model is not None:
                # Use cached prototypes if available
                proto_cache = self._get_prototype_embeddings(detector.model)
                if proto_cache:
                    import numpy as np
                    q_emb = detector.model.encode(
                        [question_lower], show_progress_bar=False,
                        normalize_embeddings=True
                    )[0]

                    best_type = None
                    best_sim = 0.35  # Minimum threshold

                    for q_type, embs in proto_cache.items():
                        for pe in embs:
                            sim = float(np.dot(q_emb, pe))
                            if sim > best_sim:
                                best_sim = sim
                                best_type = q_type

                    if best_type is not None:
                        return best_type
        except Exception:
            pass

        # Step 2: Try LLM-based classification if we have teaching types
        if self._use_llm and self._lessons.get_types():
            llm_result = self._classify_via_llm(question_lower, '')
            if llm_result:
                return llm_result

        # Step 3: Final fallback — no hardcoded word lists
        # SELF defaults to 'eksplisit' when it can't determine the type.
        # It will learn the correct type through feedback and self-correction.
        return 'eksplisit'

    def _get_prototype_embeddings(self, model) -> dict:
        """Get or compute cached prototype embeddings for each question type.

        Prototypes are the embeddings of up to 3 representative questions
        per question type. They are computed once and cached until new
        lessons are added (cache invalidation via lesson count check).

        This prevents re-encoding all prototype questions on every
        _classify_by_signals() call, which was causing timeouts.
        """
        # Check if cache is still valid
        current_lesson_count = len(self._lessons._lessons) if hasattr(self._lessons, '_lessons') else 0
        if (hasattr(self, '_proto_cache') and
                self._proto_cache is not None and
                self._proto_lesson_count == current_lesson_count):
            return self._proto_cache

        # Build new cache from teaching lessons
        proto_cache = {}
        try:
            import numpy as np
            for q_type in self._lessons.get_types():
                type_questions = []
                for lesson in self._lessons.get_by_type(q_type)[:3]:
                    if lesson.problem:
                        type_questions.append(lesson.problem)

                if not type_questions:
                    continue

                embs = model.encode(
                    type_questions, show_progress_bar=False,
                    normalize_embeddings=True
                )
                proto_cache[q_type] = [embs[i] for i in range(len(type_questions))]
        except Exception as e:
            logger.warning("Failed to build prototype embeddings cache: %s", e)
            return {}

        self._proto_cache = proto_cache
        self._proto_lesson_count = current_lesson_count
        return proto_cache

    # ═══════════════ ANSWER ROUTING ═══════════════

    def _find_answer(self, propositions, question, q_type, text):
        """Route to the appropriate answer handler based on question type.

        First checks learned patterns, then delegates to handlers.
        """
        # Step 1: Try learned patterns first
        learned = self._match_learned_pattern(text, question, q_type)
        if learned is not None:
            return learned

        # Step 2: Delegate to answer handlers
        handlers = {
            'eksplisit': self._handlers._answer_eksplisit,
            'implisit': self._handlers._answer_implisit,
            'interpretatif': self._handlers._answer_interpretatif,
            'ide_pokok': self._handlers._answer_ide_pokok,
            'peribahasa': self._handlers._answer_peribahasa,
            'bahasa_kiasan': self._handlers._answer_bahasa_kiasan,
            'teks_argumentatif': self._handlers._answer_teks_argumentatif,
            'perbandingan': self._handlers._answer_perbandingan,
            'motivasi': self._handlers._answer_motivasi,
            'sinonim_antonim': self._handlers._answer_sinonim_antonim,
            'sikap_tokoh': self._handlers._answer_sikap_tokoh,
            'teks_prosedur': self._handlers._answer_teks_prosedur,
            'teks_persuasif': self._handlers._answer_teks_persuasif,
            'unsur_cerita': self._handlers._answer_unsur_cerita,
            'benar_salah': self._handlers._answer_benar_salah,
            'analogi': self._handlers._answer_analogi,
            'kesan_pesan': self._handlers._answer_kesan_pesan,
            'penyebab_ganda': self._handlers._answer_penyebab_ganda,
            'pertanyaan_negatif': self._handlers._answer_pertanyaan_negatif,
            'tone_mood': self._handlers._answer_tone_mood,
            'teks_eksplanasi': self._handlers._answer_teks_eksplanasi,
            'inferensi_silang': self._handlers._answer_inferensi_silang,
            'konteks_makna': self._handlers._answer_konteks_makna,
        }
        handler = handlers.get(q_type)
        if handler:
            return handler(propositions, question, text)
        return {'answer': None, 'confidence': 0.0, 'method': 'unknown_type'}

    # ═══════════════ LLM FALLBACK ═══════════════

    def _should_use_llm(self, answer: dict) -> bool:
        """Check if LLM fallback should be used based on concept-based result.

        Don't use LLM if answer is already high confidence (>= 0.6).
        Use LLM if answer is None or confidence is low (< 0.5).
        """
        confidence = answer.get('confidence', 0.0)
        if confidence >= 0.6:
            return False
        return answer.get('answer') is None or confidence < 0.5

    def _try_llm_reasoning(self, text: str, question: str, previous_result: dict) -> dict:
        """Try LLM reasoning fallback via LLMReasoningEngine.

        Lazy-initializes the engine on first call. If the engine or
        any backend is unavailable, returns a dict with answer=None
        so the caller keeps the concept-based result.
        """
        try:
            llm_engine = self._get_llm_engine()
            if llm_engine is None:
                return {'answer': None, 'confidence': 0.0, 'method': 'llm_unavailable'}
            return llm_engine.reason(text, question, previous_result)
        except ImportError:
            logger.info("llm_reasoning module not available — skipping LLM fallback")
            return {'answer': None, 'confidence': 0.0, 'method': 'llm_unavailable'}
        except Exception as e:
            logger.warning("LLM reasoning fallback error: %s", e)
            return {'answer': None, 'confidence': 0.0, 'method': 'llm_error'}

    # ═══════════════ COUNTERFACTUAL VERIFICATION ═══════════════

    def _verify_via_counterfactual(self, answer: dict, text: str,
                                    propositions: list, q_type: str) -> dict:
        """Verify answer through counterfactual reasoning (proof by contradiction).

        Constructs the OPPOSITE of the answer and checks if it contradicts
        known knowledge. If the counterfactual contradicts → answer is
        more likely correct → boost confidence.
        """
        answer_val = answer.get('answer')
        if answer_val is None:
            return answer

        try:
            if not hasattr(self, '_counterfactual_verifier'):
                from derivation.counterfactual import CounterfactualVerifier
                self._counterfactual_verifier = CounterfactualVerifier(self.self_core)

            context = {
                'confidence': answer.get('confidence', 0.5),
                'propositions': propositions,
                'question_type': q_type,
            }
            verification = self._counterfactual_verifier.verify(
                str(answer_val), text, context
            )

            answer['confidence'] = verification.get('verified_confidence', answer.get('confidence', 0.5))
            answer['counterfactual_verified'] = True
            answer['counterfactual_details'] = {
                'counterfactual': verification.get('counterfactual'),
                'contradiction_found': verification.get('contradiction_found'),
                'boost': verification.get('boost_applied', 0.0),
            }

        except ImportError:
            logger.info("counterfactual module not available — skipping verification")
        except Exception as e:
            logger.warning("Counterfactual verification error: %s", e)

        return answer

    # ═══════════════ META-COGNITIVE REFLECTION ═══════════════

    def _reflect_on_failure(self, text: str, question: str, answer: dict,
                             q_type: str, propositions: list):
        """Meta-Cognitive Reflection — observe the reasoning process and identify gaps.

        Instead of blindly falling back to LLM, the system first IDENTIFIES
        what specific knowledge is missing. This information is then used
        to generate a TARGETED LLM query.
        """
        try:
            if not hasattr(self, '_meta_cognitive'):
                from derivation.meta_cognitive import MetaCognitiveMonitor
                self._meta_cognitive = MetaCognitiveMonitor(self.self_core)
                reflection_path = os.path.join(
                    os.path.dirname(__file__), '..', '..', 'data', 'reflection_log.json'
                )
                self._meta_cognitive.load_reflections(reflection_path)

            self._meta_cognitive.reflect(
                question=question,
                answer=answer,
                text=text,
                q_type=q_type,
            )

        except ImportError:
            logger.info("meta_cognitive module not available — skipping reflection")
        except Exception as e:
            logger.warning("Meta-cognitive reflection error: %s", e)

    def _try_targeted_llm_reasoning(self, text: str, question: str,
                                     previous_result: dict, q_type: str) -> dict:
        """Targeted LLM reasoning — use meta-cognitive gap identification.

        Instead of sending a generic prompt to the LLM, this method:
        1. Identifies the specific knowledge gap (via GapIdentifier)
        2. Generates a TARGETED prompt based on the gap type
        3. Sends the targeted prompt to the LLM
        """
        try:
            if not hasattr(self, '_meta_cognitive') or self._meta_cognitive is None:
                from derivation.meta_cognitive import MetaCognitiveMonitor
                self._meta_cognitive = MetaCognitiveMonitor(self.self_core)

            targeted_prompt = self._meta_cognitive.get_targeted_llm_query(
                text, question, previous_result, q_type
            )

            return self._llm_reason_with_prompt(text, question, targeted_prompt, previous_result)

        except ImportError:
            return self._try_llm_reasoning(text, question, previous_result)
        except Exception as e:
            logger.warning("Targeted LLM reasoning error: %s — falling back to generic", e)
            return self._try_llm_reasoning(text, question, previous_result)

    def _llm_reason_with_prompt(self, text: str, question: str,
                                 prompt: str, previous_result: dict) -> dict:
        """Send a custom prompt to the LLM engine.

        Used by targeted LLM reasoning to send gap-specific prompts
        instead of generic ones.
        """
        try:
            llm_engine = self._get_llm_engine()
            if llm_engine is None:
                return {'answer': None, 'confidence': 0.0, 'method': 'llm_unavailable'}

            # Use the SDK with our custom prompt
            answer = None
            if hasattr(llm_engine, '_try_sdk_fallback'):
                answer = llm_engine._try_sdk_fallback(text, question, prompt)
            if answer:
                return {
                    'answer': answer,
                    'confidence': 0.65,
                    'method': 'llm_targeted',
                    'explanation': 'Answer from targeted LLM reasoning (gap-specific prompt)',
                }

            # Try local Qwen as fallback
            if hasattr(llm_engine, '_try_local_qwen'):
                answer = llm_engine._try_local_qwen(prompt)
            if answer:
                return {
                    'answer': answer,
                    'confidence': 0.6,
                    'method': 'llm_targeted_local',
                    'explanation': 'Answer from targeted local Qwen reasoning',
                }

        except Exception as e:
            logger.warning("LLM reasoning with custom prompt failed: %s", e)

        return {'answer': None, 'confidence': 0.0, 'method': 'llm_targeted_unavailable'}

    # ═══════════════ CALIBRATION ═══════════════

    def _calibrate_confidence(self, raw: float) -> float:
        """Calibrate confidence using Platt Scaling."""
        try:
            from calibration.platt import PlattScaler
            if not hasattr(self, '_platt_scaler'):
                self._platt_scaler = PlattScaler()
                self._platt_scaler.load()
            if self._platt_scaler.fitted:
                return self._platt_scaler.calibrate(raw)
        except Exception:
            pass
        return raw

    def record_feedback(self, text: str, question: str, correct_answer: str):
        """Record feedback for confidence calibration and teaching.

        Also records the outcome as an experience episode so that
        bge-m3 retrieval can learn from this success/failure.
        Qwen3 does NOT know about the experience recording.
        """
        result = self.comprehend(text, question)
        is_correct = str(result.get('answer', '')).lower() == str(correct_answer).lower()
        raw_conf = result.get('raw_confidence', result.get('confidence', 0.0))

        try:
            if not hasattr(self, '_platt_scaler'):
                from calibration.platt import PlattScaler
                self._platt_scaler = PlattScaler()
                self._platt_scaler.load()
            self._platt_scaler.add_observation(raw_conf, is_correct)
            self._platt_scaler.save()
        except Exception:
            pass

        # Record experience episode for retrieval layer learning
        # This is IMPLICIT — Qwen3 is not aware this is happening
        if self._experience_enabled:
            self._record_experience_outcome(
                text, question, result,
                outcome='success' if is_correct else 'failure'
            )

        # Report to adaptive threshold for self-tuning
        store = self._get_experience_store()
        if store is not None and self._experience_enabled:
            method = result.get('method', '')
            node_id = self._extract_node_id_from_method(method)
            store.report_adjustment_outcome(node_id, was_correct=is_correct)

        # Also teach the correct answer
        self.teach(text, question, correct_answer)

    # ═══════════════ HELPER ═══════════════

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences, protecting Indonesian number formats.

        No hardcoded linguistic rules — just simple structural splitting.
        SELF discovers sentence boundaries through observation.
        """
        protected = re.sub(r'(\d)\.(\d{3})', r'\1_\2', text)
        protected = re.sub(r'[Rr][pP]\s*', 'RP', protected)
        sentences = re.split(r'[.!?]\s*', protected)
        result = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s = s.replace('_', '.')
            s = re.sub(r'RP', 'Rp', s)
            result.append(s)
        return result

    # ═══════════════ TEACHING ═══════════════

    def teach(self, text, question, correct_answer, explanation='',
              solution_steps=None, question_type=''):
        """Teach SELF with a structured example.

        Teacher provides:
            - Soal (question)
            - Jawaban (correct_answer)
            - Penjelasan kenapa (explanation)
            - Cara penyelesaian (solution_steps, optional)

        SELF discovers patterns through inner thinking, NOT from
        hardcoded concept clusters or keyword matching.

        Teaching flow:
            1. Create a TeachingLesson from the example
            2. Extract reasoning pattern from SELF's observations
            3. Let PatternLearner discover patterns through inner thinking
            4. Build SELF's own concept clusters from teaching
            5. Persist learned knowledge
        """
        q_type = question_type if question_type else self._classify_question(question, text)

        # ── Step 1: Create structured teaching lesson ──
        lesson = TeachingLesson(
            problem=question,
            solution_steps=solution_steps or self._infer_solution_steps(text, question, correct_answer, explanation),
            answer=correct_answer,
            explanation_why=explanation or f"Because the answer is {correct_answer}",
            question_type=q_type,
            context_text=text,
        )
        self._lessons.add(lesson)

        # ── Step 2: Store teaching example ──
        example = {
            'text': text,
            'question': question,
            'correct_answer': correct_answer,
            'explanation': explanation,
            'question_type': q_type,
        }
        self.teaching_examples.append(example)

        # ── Step 3: Extract reasoning pattern ──
        pattern = self._extract_reasoning_pattern(text, question, correct_answer, explanation, q_type)

        # ── Step 4: Build SELF's concept clusters from teaching ──
        # SELF discovers which concepts are relevant, NOT from hardcoded maps
        self._build_concept_clusters_from_teaching(text, correct_answer, q_type, pattern)

        # ── Step 5: Let PatternLearner discover patterns through inner thinking ──
        pattern_learner = self._get_pattern_learner()
        if pattern_learner is not None:
            try:
                pattern_learner.learn_from_lessons(self._lessons)
            except Exception as e:
                logger.warning("PatternLearner failed to learn from lessons: %s", e)

        # ── Step 6: Store as a learned pattern ──
        concept_paths = pattern['required_concepts']
        generalized_concepts = list(concept_paths)
        for path in concept_paths:
            parts = path.split('.', 1)
            if len(parts) == 2:
                cluster_name = parts[0]
                for sub_name in self._concept_clusters.get(cluster_name, {}):
                    sibling_path = f"{cluster_name}.{sub_name}"
                    if sibling_path not in generalized_concepts:
                        generalized_concepts.append(sibling_path)

        # Collect answer synonyms from SELF-discovered clusters
        # v35: Skip context_embeddings (dict entries, not word lists)
        answer_synonyms = set()
        answer_lower = correct_answer.lower().strip()
        for cn, subs in self._concept_clusters.items():
            for sn, words in subs.items():
                if sn == 'context_embeddings':
                    continue  # Skip embedding entries (dict, not str)
                str_words = [w for w in words if isinstance(w, str)]
                if answer_lower in [w.lower() for w in str_words]:
                    for w in str_words:
                        answer_synonyms.add(w.lower())
        answer_synonyms.add(answer_lower)

        # ── Step 6b (v36): Semantic Teaching — encode ALL data as embeddings ──
        # Instead of storing text templates, store embeddings for:
        #   - question: for semantic matching (cosine similarity)
        #   - reasoning: for LLM context when generalizing
        #   - answer: for semantic validation
        #   - key_points: for granular semantic matching
        #   - deductive_pattern: for reasoning pattern matching
        # Text is kept alongside embeddings ONLY for LLM context — never for matching.
        semantic_data = self._encode_teaching_semantic(
            question=question,
            explanation=explanation,
            correct_answer=correct_answer,
            solution_steps=solution_steps,
            text=text,
        )

        self.learned_patterns[pattern['key']] = {
            # ── Semantic data (v40): context-first matching + validation ──
            'context_embedding': semantic_data.get('context_embedding', []),
            'question_embedding': semantic_data.get('question_embedding', []),
            'reasoning_embedding': semantic_data.get('reasoning_embedding', []),
            'answer_embedding': semantic_data.get('answer_embedding', []),
            'key_points_embeddings': semantic_data.get('key_points_embeddings', []),
            'deductive_pattern_embedding': semantic_data.get('deductive_pattern_embedding', []),
            # ── Text data: ONLY for LLM context, NEVER for matching ──
            'reasoning_text': explanation or f"Taught: {pattern['key']}",
            'key_points_text': semantic_data.get('key_points_text', []),
            'deductive_pattern_text': semantic_data.get('deductive_pattern_text', ''),
            # ── Legacy compat: still stored but matching uses embeddings ──
            'answer_template': pattern['answer_template'],
            'answer': correct_answer,
            'question_type': q_type,
            'required_concepts': pattern['required_concepts'][:5],
            'generalized_concepts': generalized_concepts[:10],
            'answer_synonyms': list(answer_synonyms)[:10],
            'reasoning_method': pattern['reasoning_method'],
            'concept_signals': pattern['concept_signals'],
            'confidence': 0.6,
            'explanation': explanation or f"Taught: {pattern['key']}",
            'text_snippet': text[:100],
            'question_snippet': question[:100],
            # ── v38: System 2 Think Slow — verification & health tracking ──
            'verify_count': 0,       # Total verifications (correct + incorrect)
            'correct_count': 0,      # Number of external validations confirming correct
            'incorrect_count': 0,    # Number of external validations confirming wrong
            'last_verified': None,    # True if last validation was correct, False if wrong
            'active': True,           # False = pattern disabled (too many failures)
            'created_at': __import__('time').time(),  # When pattern was first taught
        }

        # ── Step 7: Persist learned patterns ──
        self._save_learned_patterns()

        # ── Step 8: Register in self_core if available ──
        if self.self_core:
            self.self_core.relation_registry[f'taught_{pattern["key"]}'] = {
                'count': 1,
                'examples': [f"Q: {question} A: {correct_answer}"],
                'operational_type': 'text_comprehension',
                'reasoning_method': pattern['reasoning_method'],
            }

    # ═══════════════ ORGANIC LEARNING FROM CORRECTIONS (v42) ═══════════════

    # @FLOW: CORRECTION_TEACH
    # @CALLS: _generate_correction_reasoning(), teach(), _save_learned_patterns()
    # @MUTATES: learned_patterns (via teach), self_discovered_clusters (via teach), disk (via _save_learned_patterns)
    # @BEHAVIOR: Learns from a user correction. Generates reasoning via Qwen3 to
    #            explain WHY the corrected answer is correct, then calls teach() to
    #            store the new pattern. If reasoning generation fails, still teaches
    #            with a generic explanation — never blocks learning.
    #            Returns dict with 'reasoning' (for quality gate) and 'pattern_key'.
    def teach_from_correction(self, text: str, question: str, correct_answer: str,
                               correction_raw: str = '') -> dict:
        """Learn from a user correction — organic feedback loop v1.

        v42: When a user corrects a previous answer (e.g., "salah, yang benar
        personifikasi"), this method:
          1. Generates reasoning WHY the corrected answer is correct (via Qwen3)
          2. Calls teach() to store the new pattern with the generated reasoning
          3. The pattern is persisted to disk automatically by teach()

        The generated reasoning is returned for the quality gate — the caller
        should show it to the user for confirmation. However, teach() is called
        immediately (no pending state in v1). For v2, we may add async
        confirmation where the pattern is only stored after user confirms.

        Args:
            text: The context text (narrative)
            question: The original question or the correction string
            correct_answer: The correct answer provided by the user
            correction_raw: The raw correction string (e.g., "salah, yang benar personifikasi")

        Returns:
            Dict with 'reasoning' (str) and 'pattern_key' (str).
        """
        # Step 1: Generate reasoning WHY the corrected answer is correct
        reasoning = self._generate_correction_reasoning(
            question=question,
            correct_answer=correct_answer,
            context_text=text,
        )

        # Step 2: Teach the correction as a new pattern
        # The explanation is the generated reasoning — this is what Qwen3
        # believes is the justification for why the correct answer is correct.
        explanation = reasoning or f"Koreksi pengguna: jawaban yang benar adalah {correct_answer}"

        self.teach(
            text=text,
            question=question,
            correct_answer=correct_answer,
            explanation=explanation,
        )

        # Step 3: Find the pattern key that was just created
        # The pattern key is generated by _extract_reasoning_pattern()
        # We can find it by looking for the most recently created pattern
        # with matching question and answer
        pattern_key = ''
        for pk, pd in self.learned_patterns.items():
            if (pd.get('question_snippet', '') == question[:100] and
                pd.get('answer', '') == correct_answer):
                # Mark this pattern as originating from a correction
                pd['source'] = 'correction'
                pd['correction_raw'] = correction_raw
                pattern_key = pk
                break

        # Persist the source marking
        if pattern_key:
            self._save_learned_patterns()

        logger.info(
            "v42: Learned from correction — answer=%s, reasoning=%s, pattern=%s",
            correct_answer[:50], reasoning[:80] if reasoning else 'none', pattern_key[:40]
        )

        return {
            'reasoning': reasoning,
            'pattern_key': pattern_key,
        }

    # @FLOW: CORRECTION_REASON
    # @CALLS: LLMReasoningEngine (Qwen3)
    # @MUTATES: none
    # @BEHAVIOR: Generates reasoning WHY the corrected answer is correct.
    #            Falls back to generic reasoning if LLM is unavailable.
    #            The reasoning is used for the quality gate — the user can
    #            verify that Qwen3's explanation makes sense before the
    #            pattern is stored.
    def _generate_correction_reasoning(self, question: str, correct_answer: str,
                                        context_text: str = '') -> str:
        """Generate reasoning WHY the corrected answer is correct.

        v42: Uses Qwen3 to generate an explanation for why the user's
        correction is the right answer. This reasoning serves two purposes:
          1. Quality gate: user can verify the reasoning makes sense
          2. Teaching context: stored as the pattern's explanation, so
             future similar questions benefit from the reasoning

        If Qwen3 is unavailable, falls back to a generic explanation.
        The fallback still allows learning — we never block teach() just
        because reasoning generation failed.
        """
        llm_engine = self._get_llm_engine()

        if llm_engine is not None:
            try:
                prompt = (
                    f"Seorang pengguna mengkoreksi jawaban AI.\n"
                    f"Pertanyaan: {question}\n"
                    f"Jawaban yang benar: {correct_answer}\n"
                )
                if context_text:
                    prompt += f"Teks konteks: {context_text[:500]}\n"
                prompt += (
                    f"\nJelaskan secara singkat (1-3 kalimat) MENGAPA "
                    f"jawaban '{correct_answer}' adalah jawaban yang benar "
                    f"untuk pertanyaan tersebut."
                )

                result = llm_engine.reason(
                    text=prompt,
                    question=f"Mengapa {correct_answer} adalah jawaban yang benar?",
                )
                if result and result.get('answer'):
                    return str(result['answer']).strip()
            except Exception as e:
                logger.warning("Failed to generate correction reasoning via LLM: %s", e)

        # Fallback: generic reasoning
        return f"Koreksi pengguna: jawaban yang benar untuk pertanyaan ini adalah {correct_answer}"

    def _extract_reasoning_pattern(self, text, question, correct_answer, explanation, q_type):
        """Extract the abstract reasoning pattern from a teaching example.

        SELF observes the teaching example and discovers what concepts
        and reasoning patterns are relevant. No hardcoded word lists.

        v35: Added content hash to pattern key to prevent collision.
        Two different teachings with the same q_type + top concepts
        would previously generate the same key, causing the second
        to overwrite the first. Now the key includes an 8-char hash
        of the question + answer, making each teaching unique.
        """
        # Detect concept signals in the text (from SELF-built clusters)
        concepts_detected = self._detect_concepts(text)
        concept_signals = list(concepts_detected.keys())

        # If no concepts detected, create ad-hoc signals from text
        # SELF picks important words — no hardcoded stop word lists
        if not concept_signals:
            concept_signals = self._extract_important_signals(text)[:4]

        # Determine the reasoning method
        reasoning_method = self._infer_reasoning_method(q_type, explanation)

        # Build pattern key from question type + top concepts + content hash
        # v35: Content hash prevents key collision when two teachings share
        # the same q_type and top concepts but have different answers
        top_concepts = concept_signals[:3]
        import hashlib
        content_hash = hashlib.md5(
            f"{question}:{correct_answer}".encode()
        ).hexdigest()[:8]

        if top_concepts:
            pk = f"{q_type}_{'_'.join(c.replace('.', '_') for c in top_concepts)}_{content_hash}"
        else:
            pk = f"{q_type}_generic_{content_hash}"

        # Extract answer template
        tmpl = correct_answer
        if explanation:
            for pfx in ['amanat:', 'pesan:', 'moral:', 'pelajaran:', 'makna:', 'answer:']:
                if pfx in explanation.lower():
                    idx = explanation.lower().find(pfx)
                    tmpl = explanation[idx + len(pfx):].strip().rstrip('.')
                    break

        return {
            'key': pk,
            'answer_template': tmpl,
            'required_concepts': concept_signals,
            'reasoning_method': reasoning_method,
            'concept_signals': concept_signals,
        }

    def _extract_important_signals(self, text: str) -> list:
        """Extract important signals from text — SELF discovers what matters.

        Instead of hardcoded stop word lists, SELF uses embedding similarity
        and positional heuristics to identify important words.

        This is a simplified version that picks longer, less common words.
        SELF will refine this through teaching.
        """
        words = text.lower().split()
        # Simple heuristic: longer words are more likely to be meaningful
        # SELF will refine this through teaching and pattern discovery
        scored = []
        for w in words:
            clean = w.strip('.,;:!?\'"()')
            if len(clean) > 4:
                scored.append(clean)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for w in scored:
            if w not in seen:
                seen.add(w)
                result.append(w)

        return result

    def _infer_reasoning_method(self, q_type: str, explanation: str = '') -> str:
        """Determine the reasoning method — SELF discovers through teaching.

        v32: Removed hardcoded q_type→method mapping.
        SELF discovers the appropriate reasoning method through teaching
        and observation, not from hardcoded maps.

        The method name is now inferred from the question type itself,
        or discovered through the understanding graph.
        """
        # v32: No more hardcoded mapping.
        # SELF learns the appropriate method through teaching.
        # Default: derive from question type name
        return f'{q_type}_reasoning'

    def _encode_teaching_semantic(self, question: str, explanation: str,
                                    correct_answer: str, solution_steps=None,
                                    text: str = '') -> dict:
        """Encode ALL teaching data as bge-m3 embeddings.

        v40: Added context_embedding — the embedding of the teaching context
        text. Empirical testing proved that context-to-context (c→c) is the
        most discriminative signal for subtype disambiguation. q→q cannot
        discriminate because all majas questions have near-identical embeddings.

        v36: Semantic Teaching Protocol — everything is a vector.
        Text is kept alongside embeddings ONLY for LLM context when
        generalizing to new questions. Matching is ALWAYS via cosine
        similarity in embedding space, never via keyword/text comparison.

        Encodes:
          - context_embedding: the teaching context text (v40 — PRIMARY signal)
          - question_embedding: for semantic question matching
          - reasoning_embedding: full explanation encoded for reasoning retrieval
          - answer_embedding: correct answer encoded for semantic validation
          - key_points_embeddings: each solution step encoded separately
          - deductive_pattern_embedding: the reasoning pattern (explanation + steps)

        Returns dict with embedding lists (JSON-serializable) and text for LLM context.
        """
        result = {
            'context_embedding': [],
            'question_embedding': [],
            'reasoning_embedding': [],
            'answer_embedding': [],
            'key_points_embeddings': [],
            'key_points_text': [],
            'deductive_pattern_embedding': [],
            'deductive_pattern_text': '',
        }

        try:
            from derivation.model_registry import get_shared_embedding_model
            model = get_shared_embedding_model()
            if model is None:
                logger.warning("No embedding model — semantic encoding skipped")
                return result

            # 0. Encode context text (v40 — PRIMARY matching signal)
            if text:
                ctx_emb = model.encode([text[:300]], show_progress_bar=False,
                                       normalize_embeddings=True)[0]
                result['context_embedding'] = ctx_emb.tolist()

            # 1. Encode question
            q_emb = model.encode([question], show_progress_bar=False,
                                 normalize_embeddings=True)[0]
            result['question_embedding'] = q_emb.tolist()

            # 2. Encode reasoning (full explanation)
            reasoning_text = explanation or f"The answer is {correct_answer}"
            r_emb = model.encode([reasoning_text], show_progress_bar=False,
                                 normalize_embeddings=True)[0]
            result['reasoning_embedding'] = r_emb.tolist()

            # 3. Encode answer
            a_emb = model.encode([correct_answer], show_progress_bar=False,
                                 normalize_embeddings=True)[0]
            result['answer_embedding'] = a_emb.tolist()

            # 4. Encode key points (solution steps)
            if solution_steps:
                steps_text = [str(s) for s in solution_steps if s]
                result['key_points_text'] = steps_text
                if steps_text:
                    kp_embs = model.encode(steps_text, show_progress_bar=False,
                                           normalize_embeddings=True)
                    result['key_points_embeddings'] = [e.tolist() for e in kp_embs]

            # 5. Encode deductive pattern (explanation + steps combined)
            deductive_text = reasoning_text
            if solution_steps:
                deductive_text += " | Steps: " + " → ".join(str(s) for s in solution_steps)
            dp_emb = model.encode([deductive_text], show_progress_bar=False,
                                  normalize_embeddings=True)[0]
            result['deductive_pattern_embedding'] = dp_emb.tolist()
            result['deductive_pattern_text'] = deductive_text

        except Exception as e:
            logger.warning("Semantic encoding failed: %s — text-only fallback", e)

        return result

    def _infer_solution_steps(self, text, question, correct_answer, explanation):
        """Infer solution steps from the teaching example.

        SELF discovers the solution pattern from the example itself.
        No hardcoded steps.
        """
        steps = []
        if question:
            steps.append(f"Identifikasi pertanyaan: {question[:60]}")
        if explanation:
            steps.append(f"Penjelasan: {explanation[:100]}")
        if correct_answer:
            steps.append(f"Jawaban: {correct_answer}")
        return steps or ["Observe and reason"]

    def _build_concept_clusters_from_teaching(self, text, correct_answer, q_type, pattern):
        """Build SELF's concept clusters from teaching — NO hardcoded semantic mappings.

        SELF discovers which concepts belong together through observation
        of teaching examples. Instead of hardcoded answer_concept_map,
        SELF builds clusters based on:
          1. Pattern concepts detected in the text
          2. The answer itself as a concept
          3. Context embedding for semantic matching (replaces word-list text_signals)

        v35: Replaced text_signals (generic word lists that became black holes)
        with context embeddings. Instead of storing "important words" which
        match everything, we store the teaching context as an embedding vector.
        Matching is done via cosine similarity (>= 0.5), not keyword overlap.
        This prevents the black-hole pattern where eksplisit.text_signals
        matched every question because generic long words like "menyebabkan",
        "merupakan", "berdasarkan" appeared everywhere.
        """
        # Step 1: Add pattern concepts to SELF's clusters + populate with actual words
        # v37 FIX: Previously created empty subclusters — concept matching was non-functional.
        # Now we extract meaningful words from the text that relate to each concept path
        # and populate the subcluster so _has_concept() can actually match them.
        for concept_path in pattern.get('required_concepts', []):
            parts = concept_path.split('.', 1)
            if len(parts) == 2:
                cluster_name, sub_name = parts
                if cluster_name not in self._concept_clusters:
                    self._concept_clusters[cluster_name] = {}
                if sub_name not in self._concept_clusters[cluster_name]:
                    self._concept_clusters[cluster_name][sub_name] = []

                # v37: Populate the subcluster with words from the text
                # Extract words that are semantically related to the concept path
                # Using the sub_name as a seed — look for related words in the text
                existing = set(self._concept_clusters[cluster_name][sub_name])
                # Add the sub_name itself as a concept word (it was discovered by SELF)
                sub_words = sub_name.replace('_', ' ').split()
                for w in sub_words:
                    if len(w) > 2 and w not in existing:
                        self._concept_clusters[cluster_name][sub_name].append(w)
                        existing.add(w)
                # Also add cluster_name words as concept indicators
                cluster_words = cluster_name.replace('_', ' ').split()
                for w in cluster_words:
                    if len(w) > 2 and w not in existing:
                        self._concept_clusters[cluster_name][sub_name].append(w)
                        existing.add(w)
                # Extract important words from the text that relate to this concept
                text_words = self._extract_important_signals(text)
                for w in text_words[:5]:
                    if w not in existing:
                        self._concept_clusters[cluster_name][sub_name].append(w)
                        existing.add(w)

        # Step 2: Add the answer as a concept in its own cluster
        # SELF discovers that the answer belongs to a certain concept category
        answer_lower = correct_answer.lower().strip()
        if answer_lower and len(answer_lower) > 1:
            # Create a cluster for this question type if it doesn't exist
            if q_type not in self._concept_clusters:
                self._concept_clusters[q_type] = {}

            # Add answer to a "taught_answers" subcluster
            sub_name = 'taught_answers'
            if sub_name not in self._concept_clusters[q_type]:
                self._concept_clusters[q_type][sub_name] = []
            if answer_lower not in self._concept_clusters[q_type][sub_name]:
                self._concept_clusters[q_type][sub_name].append(answer_lower)

        # Step 3: Store teaching context as embedding for semantic matching
        # v35: Instead of generic word lists (text_signals) that match everything,
        # store the actual context embedding. Matching uses cosine similarity
        # so only semantically similar contexts will match.
        try:
            from derivation.model_registry import get_shared_embedding_model
            model = get_shared_embedding_model()
            if model is not None and q_type in self._concept_clusters:
                import numpy as np
                # Encode the full teaching context (question + text snippet)
                context_to_embed = f"{text[:200]}"
                ctx_emb = model.encode(
                    [context_to_embed], show_progress_bar=False,
                    normalize_embeddings=True
                )[0]

                # Store as numpy array in a special subcluster
                sub_name = 'context_embeddings'
                if sub_name not in self._concept_clusters[q_type]:
                    self._concept_clusters[q_type][sub_name] = []
                # Store as list for JSON serialization
                self._concept_clusters[q_type][sub_name].append({
                    'embedding': ctx_emb.tolist(),
                    'answer': answer_lower,
                    'text_snippet': text[:100],
                })
        except Exception as e:
            logger.warning("Failed to store context embedding for teaching: %s", e)

    # ═══════════════ PERSISTENCE ═══════════════

    def _load_learned_patterns(self):
        """Load previously learned patterns from disk.

        SELF retains all previously taught patterns and concept clusters
        across restarts. v37: Also restores verification tracking data.

        This is the core of "System 2 Think Slow" — ALL learned knowledge
        persists across restarts. The AI doesn't forget when you restart it.
        """
        try:
            if os.path.exists(self._patterns_file):
                with open(self._patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.learned_patterns = data.get('learned_patterns', {})
                self.teaching_examples = data.get('teaching_examples', [])

                # Restore SELF-discovered concept clusters
                saved_clusters = data.get('self_discovered_clusters', {})
                for cn, subs in saved_clusters.items():
                    if cn not in self._concept_clusters:
                        self._concept_clusters[cn] = {}
                    for sn, words in subs.items():
                        if sn in self._concept_clusters[cn]:
                            existing = set(self._concept_clusters[cn][sn])
                            for w in words:
                                if w not in existing:
                                    self._concept_clusters[cn][sn].append(w)
                        else:
                            self._concept_clusters[cn][sn] = words

                # Restore teaching lessons
                lessons_data = data.get('teaching_lessons', [])
                for item in lessons_data:
                    try:
                        lesson = TeachingLesson.from_dict(item)
                        self._lessons.add(lesson)
                    except Exception:
                        pass

                # v37→v38: Backfill verification and health tracking for patterns
                for pk, pd in self.learned_patterns.items():
                    pd.setdefault('verify_count', 0)
                    pd.setdefault('last_verified', None)
                    pd.setdefault('created_at', 0)
                    # v38: Pattern health tracking
                    pd.setdefault('correct_count', 0)
                    pd.setdefault('incorrect_count', 0)
                    pd.setdefault('active', True)

                logger.info("Loaded %d learned patterns, %d teaching examples, %d concept clusters — System 2 Think Slow active",
                           len(self.learned_patterns), len(self.teaching_examples),
                           len(self._concept_clusters))

        except Exception as e:
            logger.warning("Failed to load learned patterns: %s", e)

    def provide_feedback(self, text: str, question: str,
                          predicted_answer: str, correct_answer: str,
                          answer_method: str = '', answer_confidence: float = 0.0) -> Optional[dict]:
        """Provide feedback for self-correction — EXTERNAL validation only.

        v38: This is now the ONLY place where pattern confidence changes.
        - If predicted == correct → STRENGTHEN the pattern that produced the answer
        - If predicted != correct → WEAKEN the pattern that produced the answer
        This prevents the positive feedback loop where wrong-but-confident
        patterns got stronger simply because they were used.

        v30: When the user or external system knows the correct answer,
        this triggers the full self-correction loop through AnswerHandlers.

        v34: Also records the outcome as a FAILURE experience episode
        so that bge-m3 retrieval can avoid this node in similar contexts.
        Qwen3 does NOT know about the experience recording.

        Args:
            text: Source text
            question: The question that was answered wrong
            predicted_answer: The answer SELF gave
            correct_answer: The correct answer
            answer_method: Method that produced the wrong answer (optional)
            answer_confidence: Confidence of the wrong answer (optional)

        Returns:
            Correction result dict, or None if correction not available.
        """
        predicted_str = str(predicted_answer).strip().lower()
        correct_str = str(correct_answer).strip().lower()
        is_correct = predicted_str == correct_str or correct_str in predicted_str

        if is_correct:
            # ✅ CORRECT — strengthen the pattern that produced this answer
            self._strengthen_pattern_on_validation(text, question, answer_method)
        else:
            # ❌ WRONG — weaken the pattern that produced this answer
            self._weaken_pattern_on_failure(text, question, answer_method)

            # Record failure episode for retrieval layer learning
            if self._experience_enabled:
                self._record_experience_outcome(
                    text, question,
                    {'answer': predicted_answer, 'method': answer_method,
                     'confidence': answer_confidence},
                    outcome='failure'
                )

            # Report to adaptive threshold for self-tuning
            store = self._get_experience_store()
            if store is not None and self._experience_enabled:
                node_id = self._extract_node_id_from_method(answer_method)
                store.report_adjustment_outcome(node_id, was_correct=False)

        try:
            return self._handlers.provide_feedback(
                text=text, question=question,
                wrong_answer=str(predicted_answer),
                correct_answer=str(correct_answer),
                answer_method=answer_method,
                answer_confidence=answer_confidence,
            )
        except Exception as e:
            logger.warning("Feedback processing failed: %s", e)
            return None

    def get_correction_stats(self) -> dict:
        """Get self-correction statistics.

        v30: Returns stats about SELF's self-correction performance.
        """
        try:
            return self._handlers.get_correction_stats()
        except Exception:
            return {'available': False}

    # ═══════════════ EXPERIENCE STORE ═══════════════

    def _self_evaluate_and_record(self, text: str, question: str,
                                   result: dict, q_type: str):
        """Self-evaluate answer quality and record suspected outcomes automatically.

        @FLOW:     EXPERIENCE_SELF_EVAL
        @CALLS:    ExperienceStore.record_episode() (if confidence is very low or high),
                   ExperienceStore.report_adjustment_outcome() (for threshold adaptation)
        @MUTATES:  ExperienceStore._episodes (potential append), ExperienceStore._similarity_threshold
        @BEHAVIOR: This is the AUTOMATIC feedback loop — SELF evaluates its own answer
                   after each comprehend() call and records suspected failures without
                   needing external input. This makes ExperienceWeight self-sustaining.

                   The self-evaluation uses the CALIBRATED confidence as the signal:
                   - Very low confidence (< 0.3) → suspected failure → record episode
                   - High confidence (> 0.75) → probable success → record episode
                   - Medium confidence (0.3–0.75) → uncertain → do NOT record

                   This is CONSERVATIVE by design:
                   1. We do NOT record high-confidence answers as successes unless they're
                      very confident (> 0.75). This prevents over-optimism from confident-
                      but-wrong answers.
                   2. We only record suspected failures for understanding-based answers,
                      NOT for LLM fallback answers (those have their own quality signal).
                   3. Self-evaluated episodes are marked with source='self_eval' so they
                      can be distinguished from explicit feedback.

                   Qwen3 does NOT know about this recording — it only affects bge-m3
                   retrieval scoring. The self-evaluation is purely a retrieval layer
                   mechanism that makes the experience store grow organically.
        """
        # Ablation: skip when experience recording is disabled
        if not self._experience_enabled:
            return

        store = self._get_experience_store()
        if store is None:
            return

        method = result.get('method', '')
        confidence = result.get('confidence', 0.0)
        answer = result.get('answer')

        # Don't self-evaluate LLM answers — they have their own quality signals
        if method.startswith('llm_'):
            return

        # Don't self-evaluate if answer is empty or None
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            return

        node_id = self._extract_node_id_from_method(method)

        # ── Suspected failure: very low confidence ──
        # If SELF is very unsure about its answer, it's probably wrong.
        # Record as a failure episode so future retrieval avoids this node
        # in similar contexts.
        if confidence < 0.3:
            try:
                store.record_episode(
                    context_text=f"{text} {question}",
                    node_id=node_id,
                    outcome='failure',
                    question=question,
                )
                # Report to adaptive threshold
                store.report_adjustment_outcome(node_id, was_correct=False)
                logger.debug("Self-eval: suspected failure for node %s (conf=%.2f)",
                           node_id, confidence)
            except Exception as e:
                logger.debug("Self-eval recording failed: %s", e)

        # ── Probable success: high confidence ──
        # If SELF is very confident, the answer is probably correct.
        # Record as a success episode so future retrieval boosts this node
        # in similar contexts. But be conservative — only very high confidence.
        elif confidence > 0.75:
            try:
                store.record_episode(
                    context_text=f"{text} {question}",
                    node_id=node_id,
                    outcome='success',
                    question=question,
                )
                # Report to adaptive threshold
                store.report_adjustment_outcome(node_id, was_correct=True)
                logger.debug("Self-eval: probable success for node %s (conf=%.2f)",
                           node_id, confidence)
            except Exception as e:
                logger.debug("Self-eval recording failed: %s", e)

        # ── Medium confidence: uncertain — do NOT record ──
        # We don't want to pollute the experience store with uncertain signals.
        # Only clear failures and clear successes are useful for learning.

    def _get_experience_store(self):
        """Lazy-initialize the ExperienceStore singleton.

        The ExperienceStore records success/failure episodes and provides
        experience-based adjustments to bge-m3 retrieval scores.
        Qwen3 does NOT know about this — it's purely a retrieval layer mechanism.
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

    def set_experience_enabled(self, enabled: bool):
        """Toggle experience recording and self-evaluation on/off.

        @FLOW:     EXPERIENCE_ABLATION
        @CALLS:    none — flag check only
        @MUTATES:  self._experience_enabled
        @BEHAVIOR: When disabled:
                   - _self_evaluate_and_record() becomes a no-op
                   - record_feedback() skips experience recording
                   - provide_feedback() skips experience recording
                   This is for ablation testing — does NOT clear stored data.
                   Toggle back on and the system immediately resumes recording.
        """
        self._experience_enabled = enabled

    def _record_experience_outcome(self, text: str, question: str,
                                    result: dict, outcome: str):
        """Record an experience episode for retrieval layer learning.

        @FLOW:     EXPERIENCE_RECORD
        @CALLS:    ExperienceStore.record_episode(context_text, node_id, outcome, question)
        @MUTATES:  ExperienceStore._episodes (append), experience_store.json (persist)
        @BEHAVIOR: Silently records the outcome without affecting the current answer.
                   If the result doesn't contain a usable method/node_id, the episode
                   is recorded with node_id='unknown'. Failures are not recorded if
                   outcome is not 'success' or 'failure'.

        This is the ONLY place where experience episodes are created from
        comprehension results. The ExperienceStore is also accessible from
        provide_feedback() for explicit corrections.

        Qwen3 does NOT know about this recording — it does not receive any
        context about past experiences. Only bge-m3 retrieval is affected.
        """
        store = self._get_experience_store()
        if store is None:
            return

        # Extract the node_id from the result method
        # Method format is typically: 'understanding_applied_U_xxx' or 'U_xxx_method'
        method = result.get('method', '')
        node_id = self._extract_node_id_from_method(method)

        try:
            store.record_episode(
                context_text=f"{text} {question}",  # Combine for richer context embedding
                node_id=node_id,
                outcome=outcome,
                question=question,
            )
        except Exception as e:
            logger.debug("Failed to record experience episode: %s", e)

    def _extract_node_id_from_method(self, method: str) -> str:
        """Extract understanding node ID from the answer method string.

        Method strings can be in various formats:
        - 'understanding_applied_U_signal_flip' → 'U_signal_flip'
        - 'U_quantity_compute_direct' → 'U_quantity_compute'
        - 'signal_flip_understanding' → 'signal_flip'
        - 'llm_targeted' → 'llm_targeted'

        Returns the extracted node_id, or 'unknown' if no pattern matches.
        """
        if not method:
            return 'unknown'

        # Pattern 1: 'understanding_applied_U_xxx' or 'composed_U_xxx'
        import re
        match = re.search(r'(U_[a-z_]+)', method)
        if match:
            return match.group(1)

        # Pattern 2: Known transformation kinds
        known_kinds = [
            'signal_flip', 'contrast_focus', 'negation_affirmation',
            'comparison_resolve', 'entity_extract', 'fact_extract',
            'quantity_compute', 'context_filter', 'word_sense',
        ]
        for kind in known_kinds:
            if kind in method:
                return f'U_{kind}'

        # Pattern 3: Use the method itself as identifier
        return method if len(method) < 50 else method[:50]

    def _save_learned_patterns(self):
        """Save learned patterns and SELF-discovered concept clusters to disk.

        v37: Uses raised caps (500 instead of 50) to preserve more history.
        This ensures the AI accumulates knowledge across sessions —
        the "System 2 Think Slow" principle where learning is PERMANENT.

        Uses atomic write to prevent corruption.
        """
        try:
            os.makedirs(os.path.dirname(self._patterns_file), exist_ok=True)
            data = {
                'version': 38,  # v38: System 2 Think Slow + external validation + weakening
                'learned_patterns': self.learned_patterns,
                'teaching_examples': self.teaching_examples[-self.MAX_TEACHING_EXAMPLES:],
                'self_discovered_clusters': self._concept_clusters,
                'teaching_lessons': self._lessons.to_list()[-self.MAX_TEACHING_LESSONS:],
            }
            tmp_path = self._patterns_file + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # os.replace with Windows robustness — retry after unlink on OSError
            try:
                os.replace(tmp_path, self._patterns_file)
            except OSError:
                try:
                    os.unlink(self._patterns_file)
                except OSError:
                    pass
                os.replace(tmp_path, self._patterns_file)
        except Exception as e:
            logger.warning("Failed to save learned patterns: %s", e)

    def _strengthen_pattern_on_validation(self, text: str, question: str,
                                           answer_method: str):
        """Strengthen a learned pattern when EXTERNAL validation confirms correct.

        v38: This is the ONLY way patterns get stronger — through external
        feedback confirming the answer was correct. This replaces the old
        _strengthen_pattern_if_verified() which strengthened based on confidence
        (which ≠ correctness), creating a dangerous positive feedback loop.

        Flow:
          1. Encode the question with bge-m3
          2. Find the stored pattern with highest cosine similarity
          3. If match found (sim >= 0.4): boost confidence by CONFIDENCE_BOOST_PER_VERIFY
          4. Track verification count for health monitoring
          5. Persist immediately to disk

        Qwen3 does NOT know about this — it doesn't see the confidence scores.
        """
        if not answer_method or not ('learned' in answer_method or 'semantic' in answer_method):
            return

        try:
            from derivation.model_registry import get_shared_embedding_model
            import numpy as np
            model = get_shared_embedding_model()
            if model is None:
                return

            q_emb = model.encode([question], show_progress_bar=False,
                                 normalize_embeddings=True)[0]

            best_pk = None
            best_sim = 0.0
            best_pd = None

            for pk, pd in self.learned_patterns.items():
                q_emb_stored = pd.get('question_embedding', [])
                if not q_emb_stored:
                    continue

                stored = np.array(q_emb_stored)
                norm = np.linalg.norm(stored)
                if norm > 1e-8:
                    stored = stored / norm
                sim = float(np.dot(q_emb, stored))

                if sim > best_sim:
                    best_sim = sim
                    best_pk = pk
                    best_pd = pd

            if best_pd is None or best_sim < 0.4:
                return

            # Boost confidence
            old_conf = best_pd.get('confidence', 0.6)
            new_conf = min(self.CONFIDENCE_CAP,
                          old_conf + self.CONFIDENCE_BOOST_PER_VERIFY)
            best_pd['confidence'] = new_conf

            # Track verification (correct_count)
            correct_count = best_pd.get('correct_count', 0) + 1
            best_pd['correct_count'] = correct_count
            best_pd['last_verified'] = True

            # Ensure verify_count is also updated for backward compat
            best_pd['verify_count'] = best_pd.get('verify_count', 0) + 1

            # Persist immediately
            self._save_learned_patterns()

            logger.info("✅ Pattern %s STRENGTHENED: %.3f → %.3f (correct #%d, sim=%.3f)",
                       best_pk, old_conf, new_conf, correct_count, best_sim)

        except Exception as e:
            logger.debug("Pattern strengthening failed: %s", e)

    def _weaken_pattern_on_failure(self, text: str, question: str,
                                    answer_method: str):
        """Weaken a learned pattern when EXTERNAL validation says it was WRONG.

        v38: This is the "unlearning" half of System 2 Think Slow. When
        a pattern produces a wrong answer and external feedback confirms it,
        the pattern's confidence decreases. If confidence drops below
        CONFIDENCE_INACTIVE_THRESHOLD, the pattern is marked inactive
        and will no longer be used for matching.

        This prevents patterns that consistently produce wrong answers
        from continuing to interfere with reasoning.

        Flow:
          1. Encode the question with bge-m3
          2. Find the stored pattern with highest cosine similarity
          3. If match found (sim >= 0.4): reduce confidence by CONFIDENCE_PENALTY_PER_FAILURE
          4. If confidence < CONFIDENCE_INACTIVE_THRESHOLD: mark as inactive
          5. Track failure count for health monitoring
          6. Persist immediately to disk

        Qwen3 does NOT know about this.
        """
        if not answer_method:
            return

        try:
            from derivation.model_registry import get_shared_embedding_model
            import numpy as np
            model = get_shared_embedding_model()
            if model is None:
                return

            q_emb = model.encode([question], show_progress_bar=False,
                                 normalize_embeddings=True)[0]

            best_pk = None
            best_sim = 0.0
            best_pd = None

            for pk, pd in self.learned_patterns.items():
                q_emb_stored = pd.get('question_embedding', [])
                if not q_emb_stored:
                    continue

                stored = np.array(q_emb_stored)
                norm = np.linalg.norm(stored)
                if norm > 1e-8:
                    stored = stored / norm
                sim = float(np.dot(q_emb, stored))

                if sim > best_sim:
                    best_sim = sim
                    best_pk = pk
                    best_pd = pd

            if best_pd is None or best_sim < 0.4:
                return

            # Reduce confidence
            old_conf = best_pd.get('confidence', 0.6)
            new_conf = max(self.CONFIDENCE_FLOOR,
                          old_conf - self.CONFIDENCE_PENALTY_PER_FAILURE)
            best_pd['confidence'] = new_conf

            # Track failure count
            incorrect_count = best_pd.get('incorrect_count', 0) + 1
            best_pd['incorrect_count'] = incorrect_count
            best_pd['last_verified'] = False

            # Mark inactive if confidence too low
            if new_conf < self.CONFIDENCE_INACTIVE_THRESHOLD:
                best_pd['active'] = False
                logger.warning("⚠️ Pattern %s marked INACTIVE (confidence %.3f < %.3f, failures=%d)",
                             best_pk, new_conf, self.CONFIDENCE_INACTIVE_THRESHOLD, incorrect_count)
            else:
                best_pd['active'] = True

            # Persist immediately
            self._save_learned_patterns()

            logger.info("❌ Pattern %s WEAKENED: %.3f → %.3f (failure #%d, sim=%.3f, active=%s)",
                       best_pk, old_conf, new_conf, incorrect_count, best_sim, best_pd.get('active', True))

        except Exception as e:
            logger.debug("Pattern weakening failed: %s", e)
