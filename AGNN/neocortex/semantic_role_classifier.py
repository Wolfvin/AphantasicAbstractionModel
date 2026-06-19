"""
SEMANTIC ROLE CLASSIFIER: Infer RelationType from correction text.

Biologis: BA 44 (deductive reasoning) needs typed edges to fire the
right rule. CA1's bag-of-words cue counter was a stop-gap - it could
not distinguish predicates from other tokens and had no negation
handling, so "X bukan menyebabkan Y" was tagged CAUSAL (the cue
"menyebabkan" fires) instead of DIFFERENTIAL. The SemanticRoleClassifier
closes that gap with a four-stage pipeline:

    1. SPO Parser       - extract Subject-Predicate-Object from the
                          correction text. The predicate is what
                          carries the relation semantics.
    2. Role Classifier  - map the predicate to a RelationType via seed
                          keyword tables (Indonesian + English).
    3. Negation Detector- if a negation token (bukan / tidak / not)
                          precedes the predicate, flip the result to
                          DIFFERENTIAL regardless of the seed match.
    4. Frequency Table  - every confident classification (>= one seed
                          match) bumps {predicate_normalized -> {type ->
                          count}}. Once a single type reaches the
                          override threshold (default 3), that type
                          wins over the seed rules - this is what makes
                          the classifier learnable over time.

Failure contract
----------------
- If the predicate cannot be extracted (single-word correction, empty
  text), the classifier returns RelationType.CATEGORICAL - this is the
  behaviour TrisynapticCircuit had before this component existed, so
  install/encode never crashes.
- If no seed matches and the frequency table has no entry for the
  predicate, return CATEGORICAL (same reason).
- RelationType is imported lazily from ``self-ai/src/agnn/graph.py`` so
  this module can be imported even when that tree is absent. A
  standalone ``_FallbackRelationType`` shim is provided so callers can
  still do ``SemanticRoleClassifier().classify("X menyebabkan Y")`` and
  get a member of an Enum-like object back even without the graph
  dependency. When the real ``RelationType`` is importable, the public
  ``RelationType`` symbol re-exports it (so ``classify`` returns the
  canonical enum used everywhere else).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# RelationType resolution
# ----------------------------------------------------------------------
#
# We try to import the canonical RelationType from
# self-ai/src/agnn/graph.py so the values returned by classify() are
# members of the *same* Enum that AGNNGraph / TypedEdge use. When that
# import fails (the self-ai tree is not on sys.path, e.g. during
# partial unit testing of just this module), we fall back to a local
# Enum with identical member names and values so the public API stays
# stable. The two enums are *not* interchangeable across modules, but
# callers should always go through ``SemanticRoleClassifier.classify``
# (which returns the canonical enum when available) rather than
# constructing RelationType members directly here.

def _resolve_relation_type():
    """Import RelationType from self-ai/src/agnn/graph.py.

    Adds self-ai/src to sys.path (idempotent) and returns the Enum
    class. Returns ``None`` if the module is unavailable - the caller
    then falls back to ``_FallbackRelationType``.
    """
    self_ai_src = os.path.join(
        os.path.dirname(__file__), "..", "..", "self-ai", "src"
    )
    self_ai_src = os.path.abspath(self_ai_src)
    if self_ai_src not in sys.path:
        sys.path.insert(0, self_ai_src)
    try:
        from agnn.graph import RelationType as _RT  # noqa: WPS433
        return _RT
    except Exception:
        return None


class _FallbackRelationType(Enum):
    """Local fallback when self-ai/src/agnn/graph.py is not importable.

    Values mirror the canonical RelationType so callers can compare by
    ``.value`` across the two enums.
    """
    CATEGORICAL = "categorical"
    CAUSAL = "causal"
    DIFFERENTIAL = "differential"
    FUNCTIONAL = "functional"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    DISCURSIVE = "discursive"


# Public symbol. Tests and downstream code should ``import`` this from
# here (not from agnn.graph) so the lazy fallback applies uniformly.
RelationType = _resolve_relation_type() or _FallbackRelationType


# ----------------------------------------------------------------------
# Seed keyword tables
# ----------------------------------------------------------------------
#
# Each RelationType has a set of "seed" predicates. A seed match makes
# the classification *confident* - confident calls bump the frequency
# table, and once the override threshold is reached the frequency table
# wins over the seeds (this is the learnable bit).
#
# Predicates are stored lower-cased and matched as whole-word tokens
# inside the parsed predicate slot. Multi-word seeds ("bagian dari",
# "is a", "leads to") are supported - the matcher builds an alternation
# of escaped seeds sorted longest-first so longer seeds win.

_SEED_KEYWORDS: Dict[RelationType, Tuple[str, ...]] = {
    RelationType.CAUSAL: (
        "menyebabkan", "mengakibatkan", "membuat", "memicu",
        "menghasilkan", "karena", "sehingga",
        "leads to", "causes", "caused",
    ),
    RelationType.CATEGORICAL: (
        "adalah", "merupakan", "termasuk", "tergolong",
        "bagian dari",
        "is a", "is an", "is", "are",
    ),
    RelationType.FUNCTIONAL: (
        "membutuhkan", "memerlukan", "berguna untuk",
        "berfungsi", "digunakan untuk",
        "needs", "requires",
    ),
    RelationType.TEMPORAL: (
        "setelah", "sebelum", "kemudian", "lalu", "saat",
        "ketika", "selama",
        "after", "before", "then",
    ),
    RelationType.DIFFERENTIAL: (
        "bukan", "tidak", "berlawanan", "berbeda", "kebalikan",
        "is not", "not a", "opposite",
    ),
    RelationType.DISCURSIVE: (
        "menurut", "berdasarkan", "dikatakan", "dilaporkan",
        "according to",
    ),
}


# Negation tokens. When one of these appears *before* the predicate in
# the SPO parse, the classification is flipped to DIFFERENTIAL. We keep
# this list small and explicit - the seed table for DIFFERENTIAL already
# covers the "X bukan Y" / "X tidak Y" standalone case; this list is
# specifically for the "negation + verb" pattern, e.g.
# "X tidak menyebabkan Y" -> DIFFERENTIAL (not CAUSAL).
_NEGATION_TOKENS: Tuple[str, ...] = (
    "bukan", "tidak", "not", "bukanlah", "tidaklah", "no",
)


# Override threshold: once a single RelationType accumulates this many
# counts for one predicate, the frequency table wins over the seeds.
_DEFAULT_OVERRIDE_THRESHOLD = 3


# ----------------------------------------------------------------------
# SPO parse result
# ----------------------------------------------------------------------

@dataclass
class SPO:
    """Subject-Predicate-Object parse of a correction sentence.

    Attributes:
        subject:   Leading noun phrase (everything before the predicate).
                   Empty string when the sentence is a single token.
        predicate: The matched predicate (verb / link verb). For
                   multi-word seeds ("bagian dari", "is a") the whole
                   phrase is returned. When no seed matches but a
                   candidate predicate was heuristically located (the
                   middle token of a >=3-word sentence), that candidate
                   is returned here so the frequency table can still
                   be consulted. Empty string when no predicate could
                   be located at all - in that case the classifier
                   falls back to CATEGORICAL.
        object:    Trailing noun phrase (everything after the predicate).
        raw:       The original input text (trimmed) for audit / debug.
        negated:   True when a negation token was found immediately
                   before the predicate. The classifier uses this to
                   flip the result to DIFFERENTIAL.
    """
    subject: str
    predicate: str
    object: str
    raw: str
    negated: bool = False


# ----------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------

@dataclass
class SemanticRoleClassifier:
    """Classify a correction sentence into a RelationType.

    Public API:
        classify(text) -> RelationType
        spo(text)      -> SPO                  # expose the parse

    State:
        frequency_table: {predicate_normalized: {RelationType: count}}
                          Persists across classify() calls on the same
                          instance so the classifier "learns" over
                          time. Once one type reaches
                          ``override_threshold`` for a predicate, that
                          type wins over the seed rules.

    Args:
        override_threshold: How many counts a single type must reach in
            the frequency table before it overrides the seed rules.
            Defaults to 3 (matches the spec: "count >= 3 for one type
            -> override seed rules").
    """

    override_threshold: int = _DEFAULT_OVERRIDE_THRESHOLD
    frequency_table: Dict[str, Dict[RelationType, int]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> RelationType:
        """Classify ``text`` into a RelationType.

        Pipeline:
            1. Parse SPO (seed-aware; falls back to middle-token
               heuristic for the predicate when no seed matches).
            2. If the SPO has a negation token immediately before the
               predicate, return DIFFERENTIAL (negation beats
               everything, including the frequency table - "X tidak
               menyebabkan Y" is always DIFFERENTIAL, never CAUSAL,
               even if the table says CAUSAL*100 for "menyebabkan").
            3. Look up the frequency table. If one type has reached
               ``override_threshold`` for this predicate, return it.
               This is what makes the classifier learnable - the
               table applies even when no seed matched, so a
               user-reinforced predicate can be classified without
               any seed entry.
            4. Match the predicate against the seed keyword tables.
               If a seed matches, return its RelationType and bump
               the frequency table.
            5. If no seed matches and the frequency table has no
               override, return CATEGORICAL (the historical default)
               without bumping the table - we don't want to learn
               from ambiguous inputs.

        Failure contract: any exception during parsing or matching is
        swallowed and CATEGORICAL is returned. This matches the
        no-throw contract the rest of AGNN depends on.
        """
        try:
            spo = self.spo(text)
        except Exception:
            return RelationType.CATEGORICAL

        # Step 2: negation beats everything, including the frequency
        # table. This keeps "X tidak menyebabkan Y" as DIFFERENTIAL
        # even if "menyebabkan" has been voted CAUSAL 100 times.
        if spo.negated:
            return RelationType.DIFFERENTIAL

        predicate_norm = self._normalize_predicate(spo.predicate)
        if not predicate_norm:
            return RelationType.CATEGORICAL

        # Step 3: frequency table override. Applies whether or not a
        # seed matched - this is how the classifier learns predicates
        # that were not in the seed table.
        counts = self.frequency_table.get(predicate_norm, {})
        if counts:
            dominant = max(counts, key=counts.get)
            if counts[dominant] >= self.override_threshold:
                return dominant

        # Step 4: seed match.
        seed_match = self._match_seed(spo.predicate)
        if seed_match is None:
            # Step 5: no seed match, no override -> default. We do NOT
            # bump the frequency table here because we are not
            # confident - the spec is explicit that only confident
            # (seed-match) calls populate the table.
            return RelationType.CATEGORICAL

        relation_type, _seed_text = seed_match
        # Bump the frequency table so future calls can override.
        bucket = self.frequency_table.setdefault(predicate_norm, {})
        bucket[relation_type] = bucket.get(relation_type, 0) + 1
        return relation_type

    def spo(self, text: str) -> SPO:
        """Parse ``text`` into Subject-Predicate-Object.

        Strategy:
            1. Lower-case + collapse whitespace.
            2. Build one big alternation regex of all seed phrases
               (longest first so "bagian dari" wins over "dari"). Use
               word boundaries on both sides for single-word seeds,
               and word-boundary + space on both sides for multi-word
               seeds.
            3. The first match in the text becomes the predicate.
               Everything before it is the subject, everything after
               is the object.
            4. When no seed matches, fall back to a middle-token
               heuristic: for sentences with >=3 whitespace-separated
               tokens, the middle token is treated as the predicate.
               This lets the frequency table consult still happen for
               non-seed predicates. For shorter sentences, no
               predicate is returned and the classifier defaults to
               CATEGORICAL.
            5. Walk backwards from the predicate start, scanning the
               subject tokens for any negation token. If found, set
               ``negated=True``. We only consider the *last* token
               before the predicate (the standard negation position
               in both Indonesian and English).

        Returns an SPO with empty ``predicate`` (and ``subject == raw``,
        ``object == ""``) when the input is empty or single-token -
        the classifier then falls back to CATEGORICAL.
        """
        raw = (text or "").strip()
        if not raw:
            return SPO(subject="", predicate="", object="", raw=raw)

        normalized = re.sub(r"\s+", " ", raw.lower())

        all_seeds: List[str] = []
        for seeds in _SEED_KEYWORDS.values():
            all_seeds.extend(seeds)
        # Longest first so multi-word seeds win over their single-word
        # substrings ("bagian dari" before "dari", "is a" before "is").
        all_seeds.sort(key=len, reverse=True)

        # Build one alternation pattern. Each seed is escaped and
        # wrapped so it matches as a whole phrase.
        alternation = "|".join(self._seed_pattern(s) for s in all_seeds)
        pattern = re.compile(alternation)

        # Find ALL seed matches in the text. We need to pick the "best"
        # one - this is not always the first match, because pure
        # negation tokens ("tidak", "bukan", "not") are in the seed
        # table (so "X bukan Y" classifies as DIFFERENTIAL via the
        # standalone seed path) but they also commonly *precede* a
        # real predicate in "X tidak menyebabkan Y". In that case we
        # want the real predicate ("menyebabkan") to win so the
        # negation detector can flip the result to DIFFERENTIAL via
        # the proper syntactic path (which then beats the frequency
        # table too).
        #
        # Strategy: collect all matches; if any non-negation match
        # exists, prefer the longest non-negation match (so multi-word
        # seeds win over single-word substrings). Otherwise fall back
        # to the longest match overall (covers the standalone "X bukan
        # Y" case where "bukan" *is* the predicate).
        all_matches = list(pattern.finditer(normalized))
        if not all_matches:
            # Fall through to middle-token heuristic below.
            pass
        else:
            non_negation_matches = [
                m for m in all_matches
                if m.group(0) not in _NEGATION_TOKENS
            ]
            candidate_pool = non_negation_matches or all_matches
            # Longest match wins; ties broken by earliest position
            # (stable, deterministic).
            best_match = max(
                candidate_pool,
                key=lambda m: (len(m.group(0)), -m.start()),
            )
            predicate_text = best_match.group(0)
            start, end = best_match.span()
            subject_norm = normalized[:start].strip()
            object_norm = normalized[end:].strip()
            negated = self._has_negation_before(subject_norm)
            return SPO(
                subject=subject_norm,
                predicate=predicate_text,
                object=object_norm,
                raw=raw,
                negated=negated,
            )

        # No seed matched. Fall back to middle-token heuristic so the
        # frequency table can still drive classification for non-seed
        # predicates ("mendorong", "mengakibatnya", typos, etc.).
        tokens = normalized.split(" ")
        if len(tokens) < 3:
            # Too short to extract a predicate - the classifier will
            # default to CATEGORICAL.
            return SPO(
                subject=raw, predicate="", object="", raw=raw,
                negated=False,
            )

        mid = len(tokens) // 2
        predicate_text = tokens[mid]
        subject_norm = " ".join(tokens[:mid])
        object_norm = " ".join(tokens[mid + 1:])
        negated = self._has_negation_before(subject_norm)
        return SPO(
            subject=subject_norm,
            predicate=predicate_text,
            object=object_norm,
            raw=raw,
            negated=negated,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_pattern(seed: str) -> str:
        """Build a regex fragment for one seed phrase.

        Single-word seeds use ``\\b<seed>\\b``. Multi-word seeds (those
        containing a space) use ``\\b<seed>\\b`` too - the \\b before
        the first word and after the last word suffice, and the
        internal spaces are matched literally.
        """
        escaped = re.escape(seed)
        return rf"\b{escaped}\b"

    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        """Normalize a predicate for frequency-table keying.

        Lower-cases, strips, and collapses internal whitespace. Empty
        input stays empty.
        """
        if not predicate:
            return ""
        return re.sub(r"\s+", " ", predicate.lower()).strip()

    @staticmethod
    def _has_negation_before(subject: str) -> bool:
        """True when the subject's last non-empty token is a negation.

        We deliberately only check the *last* token: in both Indonesian
        ("X tidak menyebabkan Y") and English ("X does not cause Y")
        the negation sits immediately before the predicate. Widening
        the window would risk false positives on sentences like
        "Not all humans are mortal".
        """
        if not subject:
            return False
        tokens = subject.split()
        if not tokens:
            return False
        return tokens[-1] in _NEGATION_TOKENS

    def _match_seed(
        self, predicate: str
    ) -> Optional[Tuple[RelationType, str]]:
        """Find the best seed match for ``predicate``.

        Returns ``(RelationType, seed_text)`` for the *longest* matching
        seed (so "bagian dari" beats "dari" when both are present, and
        "is a" beats "is"). Returns ``None`` when no seed matches.

        When multiple RelationTypes share the same seed length (rare -
        only happens for ambiguous single tokens), the first one in
        ``_SEED_KEYWORDS`` iteration order wins. The iteration order is
        deliberately CAUSAL, CATEGORICAL, FUNCTIONAL, TEMPORAL,
        DIFFERENTIAL, DISCURSIVE so the more "interesting" types for
        reasoning (CAUSAL / CATEGORICAL / FUNCTIONAL) win ties over the
        fallback types.
        """
        if not predicate:
            return None
        predicate_lower = predicate.lower()

        best: Optional[Tuple[RelationType, str]] = None
        best_len = -1
        for relation_type, seeds in _SEED_KEYWORDS.items():
            for seed in seeds:
                if self._seed_matches(seed, predicate_lower):
                    if len(seed) > best_len:
                        best_len = len(seed)
                        best = (relation_type, seed)
        return best

    @staticmethod
    def _seed_matches(seed: str, predicate_lower: str) -> str:
        """True when ``seed`` appears as a whole-phrase in ``predicate_lower``."""
        pattern = r"\b" + re.escape(seed) + r"\b"
        return re.search(pattern, predicate_lower) is not None
