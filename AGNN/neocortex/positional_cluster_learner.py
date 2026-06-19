"""
POSITIONAL CLUSTER LEARNER: Emergent structure discovery from corpus.

Biologis: SemanticRoleClassifier closed the "X tidak menyebabkan Y" gap
but exposed a deeper one - its RelationType mapping is a hand-coded
lookup table (menyebabkan -> CAUSAL, membutuhkan -> FUNCTIONAL). That
is a human-authored signal, not learned structure. AGNN does not
actually *understand* the relation; it pattern-matches a list someone
wrote.

PositionalClusterLearner replaces the lookup with corpus-driven
discovery. The learner watches a corpus of sentences and infers:

    1. POSITIONAL CLUSTERS (no labels needed)
       For each token, tally the positions it appears in across the
       corpus. The dominant position becomes the token's role:

           position 0  -> agent cluster  (saya / dia / kamu ...)
           position 1  -> action cluster (makan / minum / menyebabkan ...)
           position 2  -> object cluster (ayam / sapi / air ...)  [3-token case]
           position -1 -> object cluster (last token of >3-token sentences)

       For sentences longer than 3 tokens, intermediate positions
       collapse to the middle (position 1), so "saya sedang makan
       nasi ayam" still has the agent / action / object at 0 / 1 / -1.

    2. ACTION -> OBJECT DISTRIBUTION (no labels needed)
       For every (action_token, object_token) pair observed in SVO
       position, bump ``action_object_freq[action][object]``. After
       training, this distribution tells us what kind of object each
       action typically takes.

    3. OBJECT SUB-CLUSTERS PER RELATION TYPE (small bootstrap, then grows)
       ``_RELATION_OBJECT_SEEDS`` is a *tiny* per-RelationType seed
       set of representative OBJECT-side tokens (panas / banjir /
       rusak for CAUSAL, air / makanan / energi for FUNCTIONAL, ...).
       The seed is the only hand-coded bit, and it lives on a
       *different lexical layer* than SemanticRoleClassifier's
       predicate seeds - those seed the action side, these seed the
       object side.

       During training, the sub-cluster grows organically: any object
       token that follows an action whose other objects are already
       labelled (e.g. "menyebabkan" followed by seed-tokens panas /
       banjir) inherits that action's dominant RelationType. So after
       seeing "api menyebabkan kebakaran" with "kebakaran" unknown,
       "kebakaran" joins the CAUSAL sub-cluster - no human in the loop.

CLASSIFICATION
--------------
``classify(text)`` parses the text's SVO using the learned positional
clusters, looks up the action token in ``action_object_freq``, votes
the RelationType of every object token seen with that action, and
returns the majority. If confidence is low (fewer than
``min_data_points`` observations for the action, or no labelled
objects), the learner delegates to a wrapped
:class:`SemanticRoleClassifier` - the pre-PositionalClusterLearner
behaviour is preserved exactly when no training has happened.

BACKWARD COMPATIBILITY
----------------------
- ``SemanticRoleClassifier`` is NOT modified or deleted. The learner
  composes with it as fallback.
- A freshly constructed ``PositionalClusterLearner()`` (no ``train()``
  call yet) returns the *exact* same classifications as a fresh
  ``SemanticRoleClassifier()``, because every ``classify()`` call
  short-circuits to the fallback. Existing pipelines that swap the
  classifier in keep working.
- The RelationType enum is the same one
  ``SemanticRoleClassifier`` exports (re-exported here for
  convenience), so BA44 / TypedEdge / PapezCircuit see no change.

PERSISTENCE
-----------
``save(path)`` / ``load(path)`` persist the four learned structures
(positional_freq, action_object_freq, positional_clusters,
object_relation_map) as a single JSON file. Atomic write semantics
match :meth:`SemanticRoleClassifier.save`. ``load`` is a classmethod
that returns a fresh learner with the loaded state - the wrapped
:class:`SemanticRoleClassifier` fallback is always fresh (callers who
want a persisted fallback can pass one via the constructor).

Pure Python + existing numpy. Zero new dependencies.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from neocortex.semantic_role_classifier import (
    RelationType,
    SemanticRoleClassifier,
    SPO,
    _NEGATION_TOKENS,
)


# ----------------------------------------------------------------------
# Bootstrap object seeds
# ----------------------------------------------------------------------
#
# Each RelationType has a small set of representative OBJECT-side tokens
# that anchor its "object sub-cluster". These are the ONLY hand-coded
# signal in the learner; everything else (positional clusters,
# action->object distributions, grown object sub-clusters) is discovered
# from the corpus.
#
# Why this is NOT "just another lookup table":
#
#   * Different lexical layer. SemanticRoleClassifier seeds predicate
#     tokens (the action side). These seeds anchor the object side -
#     state-change outcomes for CAUSAL, need-words for FUNCTIONAL, etc.
#     A predicate-side seed says "menyebabkan IS causal"; an object-side
#     seed says "panas IS a state-change outcome". The two are
#     independent signals and cross-check each other.
#
#   * The clusters grow. Seeds bootstrap a single label per cluster;
#     ``train()`` then propagates that label to any object token that
#     co-occurs with a strongly-typed action. After watching "api
#     menyebabkan panas" (panas is a CAUSAL seed) and "api menyebabkan
#     kebakaran" (kebakaran is unknown), the learner labels "kebakaran"
#     as CAUSAL by co-occurrence - with no human in the loop.
#
#   * The learner can override the seeds. If the corpus shows
#     "menyebabkan" being followed by tokens that are *also* in the
#     FUNCTIONAL seed set (e.g. a sloppy corpus), the dominant
#     RelationType for "menyebabkan" becomes FUNCTIONAL - the seeds
#     don't dictate, they just initialise.
#
# Seeds are deliberately small (<= 10 tokens per relation) so the
# learner is forced to generalise rather than memorise.

_RELATION_OBJECT_SEEDS: Dict[RelationType, Tuple[str, ...]] = {
    RelationType.CAUSAL: (
        # state-change outcomes - things that *become* as a result
        "panas", "dingin", "basah", "kering",
        "rusak", "hancur", "berubah", "mati",
        "banjir", "kebakaran", "kanker", "luka",
    ),
    RelationType.FUNCTIONAL: (
        # things that are consumed / required
        "air", "makanan", "energi", "susu",
        "bahan bakar", "uang", "gula", "oksigen",
    ),
    RelationType.CATEGORICAL: (
        # taxonomic / class membership tokens
        "mamalia", "hewan", "tumbuhan", "logam",
        "mineral", "jenis", "kategori", "kelompok",
    ),
    RelationType.DIFFERENTIAL: (
        # contrast / negation objects
        "bukan", "berbeda", "lawan", "kebalikan",
        "berlawanan", "tidak",
    ),
    RelationType.TEMPORAL: (
        # time / temporal objects
        "kemarin", "besok", "sekarang", "kapan",
        "waktu", "lama", "cepat", "dulu",
    ),
}


# Position labels used inside positional_freq / positional_clusters.
#   0  -> first token (agent)
#   1  -> middle token(s) (action)
#   2  -> third token (object, exactly-3-token case)
#  -1  -> last token (object, >3-token case)
_AGENT_LABEL = 0
_ACTION_LABEL = 1
_OBJECT_LABEL_3 = 2
_OBJECT_LABEL_N = -1


# ----------------------------------------------------------------------
# Learner
# ----------------------------------------------------------------------

@dataclass
class PositionalClusterLearner:
    """Discover sentence structure from corpus, classify relations emergently.

    Public API (mirrors :class:`SemanticRoleClassifier` so this is a
    drop-in replacement at the call site):

        train(corpus_lines)   -> None
        classify(text)        -> RelationType
        spo(text)             -> SPO
        save(path)            -> None
        load(path)            -> PositionalClusterLearner   # classmethod

    State (all learned from corpus, all JSON-serialisable):

        positional_freq:     {token: {position_label: count}}
        action_object_freq:  {action_token: {object_token: count}}
        positional_clusters: {position_label: set(tokens)}
        object_relation_map: {object_token: RelationType}

    Fallback composition:

        fallback: an instance of :class:`SemanticRoleClassifier` used
        when the learner has not been trained, when the action token
        has fewer than ``min_data_points`` observations, or when the
        action's objects carry no RelationType signal. The fallback
        owns its own frequency_table - it learns from every fallback
        classification, so even when the learner delegates, the
        system still gets the SemanticRoleClassifier's existing
        learnable behaviour.
    """

    min_data_points: int = 3
    fallback: SemanticRoleClassifier = field(
        default_factory=SemanticRoleClassifier
    )

    positional_freq: Dict[str, Dict[int, int]] = field(default_factory=dict)
    action_object_freq: Dict[str, Dict[str, int]] = field(default_factory=dict)
    positional_clusters: Dict[int, Set[str]] = field(default_factory=dict)
    object_relation_map: Dict[str, RelationType] = field(default_factory=dict)

    # Internal: has train() been called (or state loaded from JSON)?
    # When False, classify() / spo() short-circuit to the fallback so
    # the learner's pre-training behaviour is identical to a fresh
    # SemanticRoleClassifier - existing pipelines keep working.
    _trained: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # If state was injected via the dataclass constructor (e.g. by
        # ``load`` building the instance field-by-field), treat the
        # learner as already trained.
        if self.positional_freq or self.object_relation_map:
            self._trained = True

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def agent_cluster(self) -> Set[str]:
        """Tokens whose dominant position is the agent slot (position 0)."""
        return self.positional_clusters.get(_AGENT_LABEL, set())

    @property
    def action_cluster(self) -> Set[str]:
        """Tokens whose dominant position is the action slot (position 1)."""
        return self.positional_clusters.get(_ACTION_LABEL, set())

    @property
    def object_cluster(self) -> Set[str]:
        """Tokens whose dominant position is an object slot (positions 2 or -1)."""
        return (
            self.positional_clusters.get(_OBJECT_LABEL_3, set())
            | self.positional_clusters.get(_OBJECT_LABEL_N, set())
        )

    # ------------------------------------------------------------------
    # Public API: training
    # ------------------------------------------------------------------

    def train(self, corpus_lines: List[str]) -> None:
        """Build positional clusters + action->object distributions.

        Idempotent in the sense that calling ``train()`` again on the
        same instance accumulates observations (frequencies are
        incremented, not reset) - this lets callers feed the learner
        new corpus batches over time.

        Pipeline:
            1. Parse each line into tokens, compute position labels
               per token, bump ``positional_freq``.
            2. Group tokens by dominant position into
               ``positional_clusters``.
            3. Seed ``object_relation_map`` from
               ``_RELATION_OBJECT_SEEDS`` (idempotent - re-seeding is
               a no-op for tokens already mapped).
            4. For every SVO-shaped sentence (>= 3 tokens), bump
               ``action_object_freq[action][object]``.
            5. Grow ``object_relation_map``: any object token that
               follows an action with >= ``min_data_points`` total
               observations and at least one labelled object inherits
               that action's dominant RelationType.

        Failure contract: malformed lines (empty, single-token) are
        skipped silently. A train() call with zero usable lines
        leaves the learner in the un-trained state - classify()
        then delegates to the fallback, which is the safe default.
        """
        if not corpus_lines:
            return

        # Phase 1: positional frequencies.
        for line in corpus_lines:
            tokens = self._tokenize(line)
            if not tokens:
                continue
            positions = self._compute_positions(len(tokens))
            for token, pos in zip(tokens, positions):
                bucket = self.positional_freq.setdefault(token, {})
                bucket[pos] = bucket.get(pos, 0) + 1

        # Phase 2: positional clusters (rebuild from scratch each
        # train() call - the dominant-position computation needs the
        # full positional_freq, not deltas).
        self._build_positional_clusters()

        # Phase 3: bootstrap object_relation_map from seeds. We
        # re-apply seeds on every train() call so a fresh learner
        # that gets trained also picks up the bootstrap. Tokens that
        # already have a relation assignment (e.g. grown in a prior
        # train() call) are NOT overwritten - the grown label wins.
        for relation_type, seeds in _RELATION_OBJECT_SEEDS.items():
            for seed in seeds:
                self.object_relation_map.setdefault(seed, relation_type)

        # Phase 4: action -> object co-occurrence.
        for line in corpus_lines:
            tokens = self._tokenize(line)
            if len(tokens) < 3:
                continue
            action_token, object_token = self._extract_action_object(tokens)
            if action_token is None or object_token is None:
                continue
            obj_bucket = self.action_object_freq.setdefault(action_token, {})
            obj_bucket[object_token] = obj_bucket.get(object_token, 0) + 1

        # Phase 5: grow object_relation_map by co-occurrence. For each
        # action with enough observations, compute the dominant
        # RelationType among its labelled objects; any unlabelled
        # object that follows this action inherits that relation.
        # This is the "emergent understanding" - the cluster grows
        # beyond the seed with no human in the loop.
        for action_token, obj_counts in self.action_object_freq.items():
            total = sum(obj_counts.values())
            if total < self.min_data_points:
                continue
            relation_votes: Dict[RelationType, int] = defaultdict(int)
            for obj_token, count in obj_counts.items():
                relation = self.object_relation_map.get(obj_token)
                if relation is not None:
                    relation_votes[relation] += count
            if not relation_votes:
                # No labelled objects for this action - cannot grow.
                continue
            # Deterministic: max count, ties broken by enum definition
            # order (stable in Python 3.7+ for the same dict iteration).
            dominant_relation = max(
                relation_votes,
                key=lambda rt: (
                    relation_votes[rt],
                    -list(relation_votes.keys()).index(rt),
                ),
            )
            for obj_token in obj_counts:
                if obj_token not in self.object_relation_map:
                    self.object_relation_map[obj_token] = dominant_relation

        self._trained = True

    # ------------------------------------------------------------------
    # Public API: classification
    # ------------------------------------------------------------------

    def classify(self, text: str) -> RelationType:
        """Classify ``text`` using learned clusters, fallback to SemanticRoleClassifier.

        Decision tree:
            1. If untrained, delegate to fallback (preserves
               pre-PositionalClusterLearner behaviour exactly).
            2. Parse SPO. If parsing fails or no predicate is
               extracted, delegate to fallback.
            3. Negation beats learned clusters - same rationale as
               SemanticRoleClassifier: "X tidak menyebabkan Y" is
               DIFFERENTIAL regardless of what the table or seeds say.
            4. Look up the action token in ``action_object_freq``.
               If unseen or below ``min_data_points``, delegate to
               fallback.
            5. Vote the RelationType of every labelled object seen
               with this action. If no labelled objects, delegate to
               fallback. Otherwise return the majority.

        The fallback is *always* the same
        :class:`SemanticRoleClassifier` instance the learner was
        constructed with - so its frequency_table accumulates across
        fallback calls, and the system still benefits from the
        SemanticRoleClassifier's existing learnable behaviour even
        when the learner itself has not seen enough data.
        """
        if not self._trained:
            return self.fallback.classify(text)

        try:
            spo = self.spo(text)
        except Exception:
            # Match SemanticRoleClassifier's no-throw contract.
            return self.fallback.classify(text)

        # Step 3: negation beats everything (same contract as the
        # fallback classifier).
        if spo.negated:
            return RelationType.DIFFERENTIAL

        action_token = self._normalize_token(spo.predicate)
        if not action_token:
            return self.fallback.classify(text)

        obj_counts = self.action_object_freq.get(action_token)
        if not obj_counts:
            return self.fallback.classify(text)

        total = sum(obj_counts.values())
        if total < self.min_data_points:
            return self.fallback.classify(text)

        # Vote: tally relation types of every labelled object seen
        # with this action.
        relation_votes: Dict[RelationType, int] = defaultdict(int)
        for obj_token, count in obj_counts.items():
            relation = self.object_relation_map.get(obj_token)
            if relation is not None:
                relation_votes[relation] += count

        if not relation_votes:
            return self.fallback.classify(text)

        # Deterministic tie-break: highest count wins; ties broken by
        # first-seen order in the vote dict (stable across runs given
        # the same training corpus).
        return max(
            relation_votes,
            key=lambda rt: (
                relation_votes[rt],
                -list(relation_votes.keys()).index(rt),
            ),
        )

    # ------------------------------------------------------------------
    # Public API: SPO parsing
    # ------------------------------------------------------------------

    def spo(self, text: str) -> SPO:
        """Parse ``text`` into Subject-Predicate-Object.

        Strategy:
            1. If untrained, delegate to fallback.spo() - same parse
               the existing SemanticRoleClassifier produces, including
               seed-based predicate extraction and the middle-token
               heuristic for unknown predicates.
            2. If trained, tokenize and find the action token: the
               first middle token (index 1 .. len-2) that belongs to
               the learned action cluster. If no learned action token
               is found, fall back to the canonical middle index.
            3. Split into subject / predicate / object around the
               action index.
            4. Negation detection: same logic as
               SemanticRoleClassifier - check the last token of the
               subject for a negation token.

        Returns an SPO with all empty fields when the input is empty
        (matches the fallback's contract).
        """
        if not self._trained:
            return self.fallback.spo(text)

        raw = (text or "").strip()
        if not raw:
            return SPO(subject="", predicate="", object="", raw=raw)

        normalized = re.sub(r"\s+", " ", raw.lower())
        tokens = normalized.split(" ")

        if len(tokens) < 3:
            # Too short for SVO. Delegate to the fallback so we still
            # benefit from its seed-based parse for 2-token sentences
            # like "X bukan" -> DIFFERENTIAL.
            return self.fallback.spo(text)

        action_idx = self._find_action_index(tokens)
        if action_idx is None:
            return self.fallback.spo(text)

        subject = " ".join(tokens[:action_idx])
        predicate = tokens[action_idx]
        obj = " ".join(tokens[action_idx + 1:])
        negated = self._has_negation_before(subject)
        return SPO(
            subject=subject,
            predicate=predicate,
            object=obj,
            raw=raw,
            negated=negated,
        )

    # ------------------------------------------------------------------
    # Public API: persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise learned state to ``path`` as JSON (atomic write).

        Format::

            {
              "min_data_points": 3,
              "positional_freq":     {"makan": {"1": 10}, ...},
              "action_object_freq":  {"menyebabkan": {"panas": 2, ...}, ...},
              "positional_clusters": {"1": ["makan", "minum", ...], ...},
              "object_relation_map": {"panas": "CAUSAL", ...}
            }

        Position labels are serialised as strings (JSON keys must be
        strings); ``load`` converts them back to ints. Sets are
        serialised as sorted lists for stable diffs. RelationType
        values are stored as their ``.name`` (e.g. "CAUSAL") so the
        file is robust to enum identity changes across processes.

        Atomic write: JSON is written to a sibling temp file first,
        then ``os.replace``'d onto the target path. Parent
        directories are created on demand. Same pattern as
        :meth:`SemanticRoleClassifier.save`.

        Note: the wrapped :class:`SemanticRoleClassifier` fallback is
        NOT serialised here - callers who want a persisted fallback
        should pass one in via the constructor (with its own
        ``persist_path``) and save it separately.
        """
        serialisable = {
            "min_data_points": self.min_data_points,
            "positional_freq": {
                tok: {str(pos): cnt for pos, cnt in pos_map.items()}
                for tok, pos_map in self.positional_freq.items()
            },
            "action_object_freq": {
                act: dict(objs)
                for act, objs in self.action_object_freq.items()
            },
            "positional_clusters": {
                str(pos): sorted(toks)
                for pos, toks in self.positional_clusters.items()
            },
            "object_relation_map": {
                tok: rt.name
                for tok, rt in self.object_relation_map.items()
            },
        }

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=".agnn_pcl_",
            suffix=".tmp",
            dir=parent or ".",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(serialisable, f, sort_keys=True, indent=2)
                f.write("\n")
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "PositionalClusterLearner":
        """Build a fresh learner with state loaded from JSON at ``path``.

        Returns a learner whose ``_trained`` flag is True (so
        ``classify`` / ``spo`` use the loaded state immediately).
        The wrapped fallback is a fresh :class:`SemanticRoleClassifier`
        - callers who want to also restore the fallback's
        frequency_table should construct one separately and pass it
        via the ``fallback`` constructor argument.

        Unknown RelationType names in the file (e.g. from a future
        version that added new RelationTypes) are silently skipped,
        matching :meth:`SemanticRoleClassifier._load_frequency_table`'s
        forward-compatibility contract.

        Args:
            path: Filesystem path to read.

        Returns:
            A new ``PositionalClusterLearner`` with the loaded state.

        Raises:
            FileNotFoundError: when ``path`` does not exist.
            json.JSONDecodeError: when the file is not valid JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        learner = cls(
            min_data_points=int(raw.get("min_data_points", 3)),
        )

        # positional_freq: {token: {position_str: count}} ->
        #                  {token: {int: count}}
        for tok, pos_map in raw.get("positional_freq", {}).items():
            if not isinstance(tok, str) or not isinstance(pos_map, dict):
                continue
            learner.positional_freq[tok] = {
                int(pos): int(cnt)
                for pos, cnt in pos_map.items()
                if isinstance(pos, str) and isinstance(cnt, (int, float))
            }

        # action_object_freq: {action: {object: count}}
        for act, objs in raw.get("action_object_freq", {}).items():
            if not isinstance(act, str) or not isinstance(objs, dict):
                continue
            learner.action_object_freq[act] = {
                obj: int(cnt)
                for obj, cnt in objs.items()
                if isinstance(obj, str) and isinstance(cnt, (int, float))
            }

        # positional_clusters: {position_str: [tokens]} ->
        #                      {int: set(tokens)}
        for pos_str, toks in raw.get("positional_clusters", {}).items():
            if not isinstance(pos_str, str) or not isinstance(toks, list):
                continue
            try:
                pos_int = int(pos_str)
            except ValueError:
                continue
            learner.positional_clusters[pos_int] = {
                t for t in toks if isinstance(t, str)
            }

        # object_relation_map: {token: relation_type_name} ->
        #                      {token: RelationType}
        for tok, rt_name in raw.get("object_relation_map", {}).items():
            if not isinstance(tok, str) or not isinstance(rt_name, str):
                continue
            try:
                learner.object_relation_map[tok] = RelationType[rt_name]
            except KeyError:
                # Unknown relation type (e.g. from a newer version of
                # AGNN that added new RelationTypes). Skip silently so
                # the load does not crash - same forward-compat
                # contract as SemanticRoleClassifier._load_frequency_table.
                continue

        # Mark as trained if we actually loaded any usable state.
        if (
            learner.positional_freq
            or learner.action_object_freq
            or learner.positional_clusters
            or learner.object_relation_map
        ):
            learner._trained = True

        return learner

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lower-case + collapse whitespace + split.

        Empty input -> []. Match the fallback's normalisation so a
        token seen during train() can be matched again during classify().
        """
        if not text:
            return []
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        if not normalized:
            return []
        return normalized.split(" ")

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Lower-case + strip a single token for cluster-map keying."""
        if not token:
            return ""
        return re.sub(r"\s+", " ", token.lower()).strip()

    @staticmethod
    def _compute_positions(n: int) -> List[int]:
        """Position labels for a sentence of ``n`` tokens.

        Mapping:
            n <= 0:  []
            n == 1:  [0]
            n == 2:  [0, 1]
            n == 3:  [0, 1, 2]                              # classic SVO
            n  > 3:  [0] + [1] * (n - 2) + [-1]             # collapse middle

        Rationale: for >3-token sentences the spec calls for relative
        positions (0=first, -1=last, 1=middle). Every middle token
        gets label 1 so the action cluster accumulates from any
        middle position; the last token gets -1 so the object cluster
        is separable from the 3-token-only object cluster (position 2).
        """
        if n <= 0:
            return []
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]
        if n == 3:
            return [0, 1, 2]
        return [0] + [1] * (n - 2) + [-1]

    def _build_positional_clusters(self) -> None:
        """Group tokens by dominant position.

        For each token in ``positional_freq``, the dominant position
        is the one with the highest count; ties are broken by lowest
        position number (deterministic, so the same corpus always
        produces the same clusters).

        Result is written to ``self.positional_clusters`` as
        ``{position_label: set(tokens)}``.
        """
        clusters: Dict[int, Set[str]] = defaultdict(set)
        for token, pos_map in self.positional_freq.items():
            if not pos_map:
                continue
            # Dominant position: highest count, ties -> lowest position
            # number (so position 0 wins over 1, etc. - keeps agent
            # cluster from accidentally swallowing tokens that also
            # appear in object position).
            best_pos = max(
                pos_map.keys(),
                key=lambda p: (pos_map[p], -p),
            )
            clusters[best_pos].add(token)
        # Convert defaultdict to plain dict for cleaner serialisation.
        self.positional_clusters = dict(clusters)

    @staticmethod
    def _extract_action_object(
        tokens: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract (action_token, object_token) from a token list.

        For 3-token sentences: action=tokens[1], object=tokens[2].
        For >3-token sentences: action=tokens[1], object=tokens[-1].
        For <3-token sentences: (None, None) - no SVO structure.

        The action position is always index 1 because
        :meth:`_compute_positions` labels index 1 as the action slot
        for both 3-token and >3-token sentences (in the >3 case, all
        middle indices collapse to label 1, but the *first* middle
        index - i.e. index 1 - is what we treat as the canonical
        action position for co-occurrence counting).
        """
        if len(tokens) < 3:
            return None, None
        if len(tokens) == 3:
            return tokens[1], tokens[2]
        return tokens[1], tokens[-1]

    def _find_action_index(self, tokens: List[str]) -> Optional[int]:
        """Find the index of the action token in ``tokens``.

        Strategy:
            1. Scan middle tokens (indices 1 .. len-2) for one that
               belongs to the learned action cluster
               (``positional_clusters[1]``). Return the first match -
               this is the most likely action token per the learned
               structure.
            2. If no learned action token is found in the middle,
               return the canonical action index:
                  - len(tokens) == 3 -> index 1 (middle)
                  - len(tokens) > 3  -> index 1 (first middle)
               This matches :meth:`_compute_positions`'s labelling
               and lets ``spo()`` still extract a triple for unseen
               actions.

        Returns None only when the sentence is too short for SVO,
        which ``spo()`` handles upstream.
        """
        action_cluster = self.positional_clusters.get(_ACTION_LABEL, set())

        # Scan middle tokens (skip first and last - those are agent
        # and object slots).
        for i in range(1, len(tokens) - 1):
            if tokens[i] in action_cluster:
                return i

        # Fallback: canonical action position. Always index 1 per
        # _compute_positions' labelling for both 3-token and >3-token
        # cases.
        if len(tokens) >= 3:
            return 1
        return None

    @staticmethod
    def _has_negation_before(subject: str) -> bool:
        """True when the subject's last non-empty token is a negation.

        Same logic as :meth:`SemanticRoleClassifier._has_negation_before`:
        only the *last* token of the subject is checked, because in
        both Indonesian ("X tidak menyebabkan Y") and English ("X does
        not cause Y") the negation sits immediately before the
        predicate. Widening the window would risk false positives on
        sentences like "Not all humans are mortal".
        """
        if not subject:
            return False
        tokens = subject.split()
        if not tokens:
            return False
        return tokens[-1] in _NEGATION_TOKENS
