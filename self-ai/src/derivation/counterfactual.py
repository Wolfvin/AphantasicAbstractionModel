# @WHO:   self-ai/src/derivation/counterfactual.py
# @WHAT:  Counterfactual Verification — proof by contradiction
# @PART:  self-ai/derivation
# @ENTRY: CounterfactualVerifier.verify()

"""Counterfactual Verification — Idea 2 from deductive thinking.

Core principle: "If my answer were WRONG, what would be true instead?"
If the counterfactual (opposite answer) leads to a contradiction with
known axioms, then the original answer is MORE likely correct.

This uses the EXISTING ConsistencyChecker infrastructure:
  - Negation detection
  - Antonym contradiction
  - Quantitative contradiction
  - Subject-predicate overlap

The counterfactual verifier does NOT add new hardcoded rules.
It uses the system's own contradiction detection to SELF-VERIFY.

Flow:
  1. Given answer A with confidence C
  2. Construct counterfactual: NOT A (negate the answer)
  3. Check if NOT A contradicts existing knowledge (axioms)
  4. If contradiction found → boost C (proof by contradiction)
  5. If no contradiction → reduce C (answer could be wrong)
  6. If counterfactual is CONSISTENT with knowledge → flag for review

v18: First implementation.
"""

import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class _SimpleConsistencyChecker:
    """Minimal inline consistency checker — replaces deleted consistency.checker.ConsistencyChecker.

    Provides basic contradiction detection using negation words and antonym
    matching. This is intentionally simple; SELF will learn better consistency
    checking through teaching.
    """

    # Negation words in Bahasa Indonesia
    NEGATION_WORDS = ['tidak', 'bukan', 'tanpa', 'belum', 'tak', 'tiada']

    # Fallback antonym pairs
    ANTONYM_MAP = {
        'rajin': ['malas'], 'malas': ['rajin'],
        'besar': ['kecil'], 'kecil': ['besar'],
        'tinggi': ['rendah'], 'rendah': ['tinggi'],
        'panjang': ['pendek'], 'pendek': ['panjang'],
        'banyak': ['sedikit'], 'sedikit': ['banyak'],
        'mahal': ['murah'], 'murah': ['mahal'],
        'cepat': ['lambat'], 'lambat': ['cepat'],
        'panas': ['dingin'], 'dingin': ['panas'],
    }

    def __init__(self, self_core=None):
        self.self_core = self_core

    def check_contradiction(self, text: str, axiom: dict) -> dict:
        """Check if text contradicts an axiom.

        Uses simple heuristics:
          1. Negation detection — text negates what axiom affirms (or vice versa)
          2. Antonym detection — text affirms an antonym of what axiom affirms
        """
        text_lower = text.lower()
        axiom_subj = axiom.get('subject', '').lower()
        axiom_pred = axiom.get('predicate', '').lower()
        axiom_obj = axiom.get('object', '').lower()

        # Strategy 1: Negation contradiction
        # If text contains "tidak X" and axiom says "X" (or vice versa)
        for neg in self.NEGATION_WORDS:
            pattern = rf'\b{re.escape(neg)}\b\s+(\w+)'
            for match in re.finditer(pattern, text_lower):
                negated_word = match.group(1)
                # Check if axiom affirms the negated word
                if negated_word in axiom_obj or negated_word in axiom_subj:
                    # But check the axiom doesn't also negate it
                    axiom_text = f"{axiom_subj} {axiom_pred} {axiom_obj}"
                    if not any(nw in axiom_text for nw in self.NEGATION_WORDS):
                        return {
                            'is_contradiction': True,
                            'reason': f"Text negates '{negated_word}' but axiom affirms it",
                            'confidence': 0.7,
                        }

        # Strategy 2: Antonym contradiction
        # If text affirms X and axiom affirms antonym(X)
        for word in re.findall(r'\b\w+\b', text_lower):
            antonyms = self.ANTONYM_MAP.get(word, [])
            for ant in antonyms:
                if ant in axiom_obj or ant in axiom_subj:
                    # Make sure neither is negated
                    text_has_neg = any(nw in text_lower for nw in self.NEGATION_WORDS)
                    axiom_text = f"{axiom_subj} {axiom_pred} {axiom_obj}"
                    axiom_has_neg = any(nw in axiom_text for nw in self.NEGATION_WORDS)
                    if not text_has_neg and not axiom_has_neg:
                        return {
                            'is_contradiction': True,
                            'reason': f"Text affirms '{word}' but axiom affirms antonym '{ant}'",
                            'confidence': 0.65,
                        }

        return {'is_contradiction': False, 'reason': '', 'confidence': 0.0}


class CounterfactualVerifier:
    """Verify answers through proof by contradiction.

    Instead of just trusting confidence scores, this verifier
    constructs the OPPOSITE of the answer and checks if it
    contradicts known knowledge. If it does, the original
    answer gains confidence (proof by contradiction).

    This is a meta-cognitive tool: the system questions its
    own answers by testing their negations.
    """

    # Boost applied when counterfactual contradicts knowledge
    CONTRADICTION_BOOST = 0.12
    # Penalty when counterfactual is consistent (answer might be wrong)
    CONSISTENCY_PENALTY = 0.08
    # Maximum confidence boost from counterfactual verification
    MAX_BOOST = 0.20

    def __init__(self, self_core=None):
        self.self_core = self_core
        self._consistency_checker = None
        self._verification_log = []  # Track verifications for learning

    def _get_consistency_checker(self):
        """Lazy init ConsistencyChecker — replaced with inline simple checker."""
        # consistency.checker was deleted; use inline simple consistency checks instead.
        # The SimpleConsistencyChecker provides basic negation/antonym contradiction detection.
        if self._consistency_checker is None:
            self._consistency_checker = _SimpleConsistencyChecker(self.self_core)
        return self._consistency_checker

    def verify(self, answer: str, text: str, context: dict = None) -> dict:
        """Verify an answer through counterfactual reasoning.

        Args:
            answer: The proposed answer to verify
            text: The source text the answer was derived from
            context: Additional context (question type, propositions, etc.)

        Returns:
            dict with keys:
                verified_confidence: float — adjusted confidence
                counterfactual: str — the negated answer
                contradiction_found: bool — whether counterfactual contradicts
                contradiction_type: str — which strategy found the contradiction
                boost_applied: float — confidence adjustment
        """
        context = context or {}

        # Step 1: Construct the counterfactual (negation of the answer)
        counterfactual = self._construct_counterfactual(answer, context)

        if counterfactual is None:
            # Cannot construct counterfactual → neutral verification
            return {
                'verified_confidence': context.get('confidence', 0.5),
                'counterfactual': None,
                'contradiction_found': False,
                'contradiction_type': 'no_counterfactual',
                'boost_applied': 0.0,
            }

        # Step 2: Check if counterfactual contradicts known axioms
        contradiction_result = self._check_counterfactual_contradiction(
            counterfactual, text, context
        )

        # Step 3: Adjust confidence based on result
        base_confidence = context.get('confidence', 0.5)
        boost = 0.0

        if contradiction_result.get('is_contradiction'):
            # Counterfactual contradicts → original answer is MORE likely correct
            boost = min(self.CONTRADICTION_BOOST, self.MAX_BOOST)
            # Higher confidence contradiction → bigger boost
            contradiction_conf = contradiction_result.get('confidence', 0.5)
            boost *= contradiction_conf  # Scale by contradiction confidence
        else:
            # Counterfactual is consistent → original answer might be wrong
            # Only penalize if we have enough axioms to check against
            if self._has_sufficient_knowledge(context):
                boost = -self.CONSISTENCY_PENALTY

        verified_confidence = max(0.0, min(1.0, base_confidence + boost))

        # Step 4: Log for meta-cognitive learning
        verification_record = {
            'answer': answer[:100],
            'counterfactual': counterfactual[:100],
            'contradiction_found': contradiction_result.get('is_contradiction', False),
            'contradiction_type': contradiction_result.get('reason', '')[:80],
            'boost': boost,
            'base_confidence': base_confidence,
            'verified_confidence': verified_confidence,
        }
        self._verification_log.append(verification_record)

        return {
            'verified_confidence': verified_confidence,
            'counterfactual': counterfactual,
            'contradiction_found': contradiction_result.get('is_contradiction', False),
            'contradiction_type': contradiction_result.get('reason', 'none'),
            'boost_applied': boost,
        }

    def _construct_counterfactual(self, answer: str, context: dict) -> Optional[str]:
        """Construct the counterfactual (negation) of an answer.

        Strategies:
          1. For qualitative answers: negate with "tidak" / "bukan"
          2. For antonym-based answers: replace with antonym
          3. For quantitative answers: use a different number
          4. For boolean answers: flip True/False
        """
        answer_lower = answer.lower().strip()

        # Strategy 1: Boolean flip
        if answer_lower in ('benar', 'ya', 'betul', 'tepat'):
            return 'salah'
        if answer_lower in ('salah', 'tidak', 'bukan', 'tidak benar'):
            return 'benar'

        # Strategy 2: Antonym flip
        antonym = self._find_antonym(answer_lower)
        if antonym:
            return antonym

        # Strategy 3: Negation with "tidak" or "bukan"
        # Check if answer already contains negation
        negation_words = ['tidak', 'bukan', 'tanpa', 'belum', 'tak', 'tiada']
        has_negation = any(nw in answer_lower.split() for nw in negation_words)

        if has_negation:
            # Remove negation to get counterfactual
            for nw in negation_words:
                # Handle double negation: "tidak bukan" → remove both
                pattern = rf'\b{re.escape(nw)}\b\s*'
                counterfactual = re.sub(pattern, '', answer_lower, count=1).strip()
                if counterfactual and counterfactual != answer_lower:
                    return counterfactual
        else:
            # Add negation
            # Choose "bukan" for nouns, "tidak" for adjectives/verbs
            if self._is_noun_phrase(answer_lower):
                return f'bukan {answer_lower}'
            else:
                return f'tidak {answer_lower}'

        # Strategy 4: Quantitative counterfactual
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            # Replace each number with a different value
            counterfactual = answer
            for num in numbers:
                val = float(num)
                # Use a nearby but different value
                alt_val = val + 1 if val < 100 else val * 1.5
                if alt_val == int(alt_val):
                    alt_val = int(alt_val)
                counterfactual = counterfactual.replace(num, str(alt_val), 1)
            return counterfactual

        return None

    # Small fallback antonym map — SELF will build a richer one through teaching.
    _FALLBACK_ANTONYM_MAP = {
        'rajin': ['malas'], 'malas': ['rajin'],
        'besar': ['kecil'], 'kecil': ['besar'],
        'tinggi': ['rendah'], 'rendah': ['tinggi'],
        'panjang': ['pendek'], 'pendek': ['panjang'],
        'banyak': ['sedikit'], 'sedikit': ['banyak'],
        'mahal': ['murah'], 'murah': ['mahal'],
        'cepat': ['lambat'], 'lambat': ['cepat'],
        'panas': ['dingin'], 'dingin': ['panas'],
        'terang': ['gelap'], 'gelap': ['terang'],
        'senang': ['sedih'], 'sedih': ['senang'],
        'baik': ['buruk'], 'buruk': ['baik'],
    }

    def _find_antonym(self, word: str) -> Optional[str]:
        """Find an antonym for a word using fallback antonym map."""
        # Try CONCEPT_CLUSTERS first (if available in future)
        try:
            from derivation.concepts import CONCEPT_CLUSTERS
            antonym_map = CONCEPT_CLUSTERS.get('antonym_map', {})
            if word in antonym_map:
                return antonym_map[word][0]
            for key, values in antonym_map.items():
                if word in values:
                    return key
        except ImportError:
            pass

        # Fallback to built-in antonym map
        if word in self._FALLBACK_ANTONYM_MAP:
            return self._FALLBACK_ANTONYM_MAP[word][0]
        return None

    def _is_noun_phrase(self, text: str) -> bool:
        """Heuristic: is the text likely a noun phrase?

        In Bahasa Indonesia, "bukan" negates nouns, "tidak" negates
        adjectives/verbs. This heuristic uses common Indonesian
        adjective/verb prefixes to distinguish.
        """
        # Common Indonesian verb/adjective prefixes
        verb_adj_prefixes = ['ber', 'me', 'di', 'ter', 'pe', 'se', 'per']
        words = text.split()
        if not words:
            return False

        first_word = words[0]
        for prefix in verb_adj_prefixes:
            if first_word.startswith(prefix) and len(first_word) > len(prefix) + 1:
                return False  # Likely verb/adjective → use "tidak"

        return True  # Likely noun → use "bukan"

    def _check_counterfactual_contradiction(self, counterfactual: str,
                                             text: str, context: dict) -> dict:
        """Check if the counterfactual contradicts existing knowledge.

        Uses ConsistencyChecker to test the counterfactual against:
          1. Axioms from the knowledge store
          2. Facts extracted from the source text
        """
        checker = self._get_consistency_checker()

        # Check against axioms if available
        if self.self_core and hasattr(self.self_core, 'axioms'):
            for axiom_id, axiom in self.self_core.axioms.items():
                result = checker.check_contradiction(counterfactual, axiom)
                if result.get('is_contradiction'):
                    return result

        # Check against propositions from the text
        propositions = context.get('propositions', [])
        for prop in propositions:
            # Build a pseudo-axiom from the proposition
            pseudo_axiom = {
                'subject': prop.get('subject', ''),
                'predicate': prop.get('predicate', ''),
                'object': prop.get('object', ''),
            }
            result = checker.check_contradiction(counterfactual, pseudo_axiom)
            if result.get('is_contradiction'):
                return result

        # Check against the source text itself
        # If the counterfactual directly contradicts a statement in the text
        text_contradiction = self._check_text_contradiction(counterfactual, text)
        if text_contradiction.get('is_contradiction'):
            return text_contradiction

        return {'is_contradiction': False, 'reason': '', 'confidence': 0.0}

    def _check_text_contradiction(self, counterfactual: str, text: str) -> dict:
        """Check if counterfactual directly contradicts a statement in the source text.

        This is a lightweight check that looks for:
          1. The counterfactual negates something the text affirms
          2. The counterfactual affirms something the text negates
        """
        text_lower = text.lower()
        cf_lower = counterfactual.lower()

        # Check if text affirms what counterfactual negates
        # E.g., text says "rajin" and counterfactual is "tidak rajin"
        negation_words = ['tidak', 'bukan', 'tanpa', 'belum', 'tak', 'tiada']

        for neg in negation_words:
            # Pattern: counterfactual says "tidak X" and text says "X"
            neg_pattern = rf'\b{re.escape(neg)}\b\s+(\S+)'
            for match in re.finditer(neg_pattern, cf_lower):
                negated_word = match.group(1)
                # If the negated word appears positively in the text
                if re.search(rf'\b{re.escape(negated_word)}\b', text_lower):
                    # And the negation doesn't appear near it in text
                    word_pos = text_lower.find(negated_word)
                    nearby = text_lower[max(0, word_pos - 20):word_pos + len(negated_word) + 5]
                    if not any(nw in nearby for nw in negation_words):
                        return {
                            'is_contradiction': True,
                            'reason': f"Counterfactual '{counterfactual}' contradicts text which affirms '{negated_word}'",
                            'confidence': 0.75,
                        }

        return {'is_contradiction': False, 'reason': '', 'confidence': 0.0}

    def _has_sufficient_knowledge(self, context: dict) -> bool:
        """Check if there's enough knowledge to make counterfactual
        verification meaningful.

        If we have no axioms or propositions, the absence of a
        contradiction doesn't mean much — we just don't know enough.
        """
        has_axioms = (self.self_core is not None and
                      hasattr(self.self_core, 'axioms') and
                      len(self.self_core.axioms) > 0)
        has_propositions = len(context.get('propositions', [])) > 0
        return has_axioms or has_propositions

    def get_verification_stats(self) -> dict:
        """Get statistics about counterfactual verifications.

        This is used by the meta-cognitive monitor to understand
        how often answers pass counterfactual verification.
        """
        if not self._verification_log:
            return {'total': 0, 'contradiction_rate': 0.0, 'avg_boost': 0.0}

        total = len(self._verification_log)
        contradictions = sum(1 for v in self._verification_log if v['contradiction_found'])
        avg_boost = sum(v['boost'] for v in self._verification_log) / total

        return {
            'total': total,
            'contradictions': contradictions,
            'contradiction_rate': contradictions / total if total > 0 else 0.0,
            'avg_boost': avg_boost,
        }
