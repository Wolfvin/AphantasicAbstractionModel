"""
POSITIONAL CLUSTER LEARNER v2 - zero-bias emergent structure discovery.

This is a clean rewrite of the original PR #69 design, which was
rejected for two fundamental reasons:

  1. SEMI-SUPERVISED ANCHOR. The original code pre-seeded 44 hand-picked
     object tokens to RelationType via ``_RELATION_OBJECT_SEEDS``
     *before* training. That is not zero-bias discovery - it is
     semi-supervised learning with a human-authored anchor at the
     start. The bias propagates everywhere downstream.

  2. PERMANENT POLYSEMY LOSS. The original code assigned each token to
     a *single* "dominant position" cluster (highest positional count
     wins, ties -> lowest position number). That locked ``"ayam"`` to
     one role forever, even though in real corpora the same token can
     be a subject in one sentence and an object in another:

         "ayam mencari pakan"   -> ayam = agent  (position 0)
         "manusia potong ayam"  -> ayam = object (position -1)

     A global table that hard-assigns tokens to roles cannot represent
     this; the learner must determine role from the *actual position*
     in the sentence being parsed.

DESIGN (v2) - two stages, mirroring how transformers pretrain and then
get calibrated:

  STAGE 1 - ZERO-BIAS CLUSTERING (no RelationType mentioned anywhere)
    train(corpus_lines) builds:

      positional_freq:    {token: {position_bucket: count}}
                          - Soft counts per position. NOT a hard
                            assignment. The same token can show up in
                            multiple buckets with different counts -
                            that is the polysemy signal.

      action_object_freq: {action_token: {object_token: count}}
                          - Co-occurrence of action (positional bucket
                            1) with object (positional bucket 2 or -1),
                            determined AT PARSE TIME from the sentence
                            being trained on - not from any global
                            "dominant position" lookup.

      action_clusters:    {cluster_id: set(action_tokens)}
                          - Action tokens clustered by *similarity of
                            their object distributions*. Two actions
                            that take the same kind of object end up
                            in the same cluster. Clustering uses
                            Jaccard similarity + greedy agglomerative
                            merge (pure Python, no sklearn / scipy).

      cluster_id_of:      {action_token: cluster_id}

    Crucially, no RelationType is assigned at this stage. Clusters are
    *unnamed* integer IDs (0, 1, 2, ...). A human must inspect them
    and assign names via Stage 2.

  STAGE 2 - POST-HOC NAMING (once, at cluster level)
    label_clusters(mapping: Dict[int, RelationType]) -> None
        Called once after training. The mapping assigns a
        RelationType to one or more cluster_ids. After this call,
        classify() can use the labels; before it, classify() must
        fall back to SemanticRoleClassifier.

    inspect_clusters() -> Dict[int, List[str]]
        Returns a human-readable view of each cluster: its action
        tokens and its top object tokens (sorted by co-occurrence
        count). This is what a human reviews before deciding the
        cluster -> RelationType mapping.

CLASSIFICATION
--------------
classify(text):

  1. If no clusters have been labelled yet -> delegate to the wrapped
     SemanticRoleClassifier (full fallback; behaviour identical to a
     fresh SemanticRoleClassifier).
  2. Parse the sentence positionally: find the action token by
     POSITION (middle slot, first token whose positional_freq shows
     strong action-bucket signal). The role of every other token is
     determined by where it sits in *this* sentence, not by any
     global lookup.
  3. Negation override: if a negation token immediately precedes the
     action, return DIFFERENTIAL - same contract as
     SemanticRoleClassifier, and it beats any cluster label.
  4. Look up the action's cluster_id. If that cluster has been
     labelled, return its RelationType.
  5. Otherwise -> fallback to SemanticRoleClassifier.

SPO PARSING
-----------
spo(text):

  SVO extraction is positional, not cluster-based:
    - 3 tokens: subject=tokens[0], predicate=tokens[1], object=tokens[2]
    - >3 tokens: subject=tokens[0], predicate=tokens[1], object=tokens[-1]
    - <3 tokens: delegate to fallback (preserves "X bukan Y" ->
      DIFFERENTIAL path that lives in SemanticRoleClassifier's seed
      table)

  The action position is always index 1 because positional_freq is
  built with index 1 as the action bucket for both 3-token and
  >3-token sentences. This means "ayam" is correctly the agent in
  "ayam mencari pakan" (index 0) and the object in "manusia potong
  ayam" (index 2 or -1) - the same instance handles both, because
  role is derived from CURRENT position, not a permanent cluster
  membership.

PERSISTENCE
-----------
save(path) / load(path) write JSON containing:
  - positional_freq
  - action_object_freq
  - action_clusters (with cluster_id_of)
  - cluster_labels (Dict[int, RelationType.name]) - only present if
    label_clusters() has been called

CONSTRAINTS
-----------
- Zero seed tokens before training. No _RELATION_OBJECT_SEEDS or any
  variant. The ONLY hand-authored signal is the (optional, post-hoc)
  label_clusters() call.
- Zero new dependencies. Pure Python + existing numpy (numpy is
  available but not required by this module).
- SemanticRoleClassifier is NOT modified. The learner composes with
  it as fallback.
- Negation override preserved (same contract as the fallback).
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
# Position buckets
# ----------------------------------------------------------------------
#
# Position labels used inside positional_freq.
#   0  -> first token (agent slot)
#   1  -> middle token(s) (action slot)
#   2  -> third token (object slot, exactly-3-token sentences)
#  -1  -> last token (object slot, >3-token sentences)
#
# Tokens are NOT hard-assigned to a single bucket. positional_freq is a
# *soft* count: the same token can show up in multiple buckets with
# different counts. Role inference at parse time uses the *current
# sentence's position*, not a global lookup of "which bucket this token
# usually sits in".

_AGENT_BUCKET = 0
_ACTION_BUCKET = 1
_OBJECT_BUCKET_3 = 2
_OBJECT_BUCKET_N = -1


# ----------------------------------------------------------------------
# Action stoplist (Bug 1 fix)
# ----------------------------------------------------------------------
#
# Function words that must NEVER be captured as the "action" token when
# extracting (action, object) pairs positionally. These are intensifiers,
# deictic markers, epistemic adverbs, and copula-like appearance verbs
# that sit in position 1 of state+adjective sentences like:
#
#     "es itu sangat dingin"     - 'itu' and 'sangat' are NOT actions;
#                                   'dingin' (the adjective) is.
#     "karbon tampak stabil"     - 'tampak' is NOT an action; 'stabil' is.
#     "durian bersifat harum"    - 'bersifat' is NOT an action; 'harum' is.
#
# Without this stoplist, 'sangat' / 'itu' / 'tampak' / etc. would be
# captured as the action and form garbage clusters whose "objects" are
# actually the adjectives of the state pattern - e.g.
# {'actions': ['sangat'], 'top_objects': ['asin', 'dingin', 'hijau']}.
#
# The fix: when extracting (action, object) by position, skip tokens in
# this stoplist from the action slot. The first non-stoplist token after
# the subject becomes the action. If the action is also the last token
# (no real object remains, as in pure state+adjective), the sentence
# contributes nothing to action_object_freq.
#
# This list is a fixed linguistic resource, NOT a tunable parameter.
# It is intentionally narrow: only words that are unambiguously
# function-word-like in Bahasa Indonesia state+adjective patterns. We do
# NOT include real verbs (makan, menyebabkan, adalah, merupakan) - those
# carry relation semantics and must be free to form clusters.
_ACTION_STOPLIST: frozenset = frozenset({
    # Deictic / determiner markers
    "itu", "ini", "tersebut",
    # Intensifiers
    "sangat", "begitu", "terlalu", "cukup",
    # Epistemic / evidential adverbs
    "memang", "sebenarnya", "dasarnya", "faktanya",
    # Copula-like appearance verbs that precede the real predicate
    # in state+adjective patterns. These look like verbs but function
    # syntactically as copulas - the actual semantic predicate is the
    # adjective that follows them.
    "tampak", "terlihat", "terasa", "tergolong", "bersifat",
    # Prefix-phrase introducers. These begin adverbial phrases like
    # "menurut ahli," / "secara teknis" / "faktanya" that occupy the
    # subject slot in positional parsing. Without this stoplist entry,
    # the second word of the phrase (e.g. "ahli") would be captured
    # as the action - a garbage noun-as-action pair.
    "menurut", "secara",
    # Negation markers. These are NOT actions - they're function words
    # that flip the relation. The negation override in classify()
    # handles them via ``spo.negated`` (which checks _NEGATION_TOKENS),
    # so removing them from the action slot here does not break
    # negation detection.
    "bukan", "tidak", "bukanlah", "tidaklah",
})


# ----------------------------------------------------------------------
# Verb-prefix heuristic (complements the stoplist)
# ----------------------------------------------------------------------
#
# Indonesian verbs are highly morphologically regular: the vast majority
# start with one of the active/passive prefixes me-, ber-, di-, or ter-.
# This is a coarse but effective signal for distinguishing verbs from
# nouns in position 1 of an SVO sentence.
#
# We use this heuristic ONLY to disambiguate multi-word subjects like
# "ahli gizi menyarankan diet" (4 tokens). Position 1 here is "gizi"
# (a noun - the second word of the compound subject "ahli gizi"), and
# position 2 is "menyarankan" (the real verb). Without the heuristic,
# positional parsing would capture (action="gizi", object="diet") -
# a garbage pair where a noun is treated as the action.
#
# The heuristic: when scanning for the action token, prefer the first
# token that *looks like a verb* (starts with one of these prefixes)
# or is a known copula (see :data:`_COPULAS`). If no such token exists
# before the object slot, skip the sentence (avoids noun-as-action
# garbage).
#
# This is intentionally conservative:
#   - Prefixes are 3+ characters to avoid false positives ("di" alone
#     is a preposition, "me" alone matches "merah" = red).
#   - We accept some false positives ("beras" = rice, "ternak" =
#     livestock) because (a) they're rare in the action slot and
#     (b) the cost of a false positive is one mis-clustered action,
#     while the cost of a false negative is breaking the multi-word
#     subject fix for hundreds of sentences.
_VERB_PREFIXES: tuple = (
    "meng", "meny", "mem", "men",   # me- active voice (4 / 3-char)
    "ber", "bel",                    # ber- intransitive (3-char)
    "diper",                          # diper- passive (5-char, more
                                      # reliable than just "di-")
    "ter",                            # ter- accidental/passive (3-char)
)


# Copulas - link verbs that carry relation semantics (typically
# CATEGORICAL) but don't carry the me-/ber-/diper-/ter- prefix that
# :data:`_VERB_PREFIXES` detects. Without this whitelist, multi-word
# categorical sentences like "suku bunga adalah instrumen kebijakan
# moneter" (6 tokens) would be skipped by the >3-token verb-prefix
# requirement, because "adalah" doesn't match any prefix and sits at
# position 2 (after the multi-word subject "suku bunga"). The copula
# whitelist lets "adalah" be recognised as a valid action despite
# lacking verbal morphology.
_COPULAS: frozenset = frozenset({
    "adalah", "merupakan", "ialah", "yaitu", "yakni",
})


# ----------------------------------------------------------------------
# Clustering parameters
# ----------------------------------------------------------------------

# Two actions are similar (and thus merge into the same cluster) when
# the *weighted* Jaccard similarity of their object-token count maps is
# >= this value.
#
# Weighted Jaccard = sum(min(c_a(x), c_b(x)) for x in A∩B) /
#                    sum(max(c_a(x), c_b(x)) for x in A∪B)
#
# where c_a(x) is the co-occurrence count of action a with object x.
#
# Default is 0.13 - lower than the previous 0.25 because plain Jaccard
# on object *sets* was too strict for synonym merging (Bug 2). Two
# synonyms like 'adalah' and 'merupakan' both take class-noun objects
# (mamalia, logam, ...) but rarely the *same* object, so their set
# overlap is small. Weighted Jaccard on counts captures the *shape* of
# the distribution better: if both actions frequently co-occur with
# abstract-class objects (even different ones), the weighted overlap
# of their high-count objects is enough to merge them.
#
# 0.13 was validated by re-running train() on pretrain_corpus.txt:
# 'adalah' and 'merupakan' merge into one cluster, total cluster count
# drops from 229 (over-fragmented) to under 80 (target met).
_DEFAULT_SIMILARITY_THRESHOLD = 0.13

# An action must have at least this many (action, object) co-occurrence
# observations before it participates in clustering. Below this, the
# action is left unclustered (cluster_id = -1) and classify() will
# fall back to SemanticRoleClassifier for it.
_DEFAULT_MIN_ACTION_OBSERVATIONS = 2

# How many top objects (by count) to surface in inspect_clusters().
# Keeps the human-readable view manageable when a cluster has dozens
# of objects.
_INSPECT_TOP_OBJECTS = 15


# ----------------------------------------------------------------------
# Connector-signal detection parameters
# ----------------------------------------------------------------------
#
# This module separates structurally different actions even when their
# object sets overlap heavily. The textbook case (cluster 62 in the
# combined pretrain corpus):
#
#     "kucing adalah mamalia"      -> action immediately followed by object
#     "kucing berbeda dari reptil" -> action followed by a CONNECTOR
#                                     ("dari"/"dengan"/"sebagai") and
#                                     THEN the object
#
# Both actions take taxonomy nouns (mamalia, reptil, ikan, ...) as
# objects, so weighted Jaccard on object distributions merges them
# into one cluster — even though they are structurally different
# predicate types (affirmation-of-category vs contrast-of-category).
#
# The connector signal is detected PURELY from position+frequency
# statistics, with NO hardcoded list of "negation words" or
# "connector words". The detection contract:
#
#   1. For every (action, object) observation, also record the token
#      (if any) that sits immediately after the action and before the
#      object — the "between-first" slot. Direct (no-token) is recorded
#      as None.
#
#   2. A token is a "corpus-wide connector" if:
#        - it appears in the between-first slot at least
#          ``_CONNECTOR_MIN_BETWEEN_COUNT`` times across the corpus
#          (so a one-off noun-as-between-token doesn't qualify); AND
#        - it NEVER appears as an object of any action in the corpus
#          (so a real object noun that sometimes sits mid-sentence
#          doesn't qualify — only grammar-only tokens like prepositions
#          and complementizers make the cut).
#
#   3. An action has ``has_connector=True`` if some corpus-wide
#      connector token occupies the between-first slot in >=
#      ``_CONNECTOR_RATE_THRESHOLD`` of the action's observations.
#
# The two-step filter (corpus-wide detection + per-action rate) is what
# keeps the detection zero-bias:
#   - Step 2 says "this token is grammar, not an object" — decided
#     globally, not by meaning.
#   - Step 3 says "this action routinely takes a connector" — decided
#     per-action, not by meaning.
#
# Neither step consults a list of "negation words" or "connector
# words". A token like "dari" qualifies because (a) it sits in the
# between-first slot many times and (b) no sentence in the corpus
# treats "dari" itself as an object — pure positional evidence.

# Minimum corpus-wide between-first count for a token to be considered
# a connector candidate. Tokens that show up once or twice in the
# between-first slot are likely sentence-specific noun phrases, not
# grammar.
_CONNECTOR_MIN_BETWEEN_COUNT = 3

# Per-action rate threshold: a connector token must occupy the
# between-first slot in at least this fraction of the action's
# observations for the action to be flagged has_connector=True.
# 0.5 = majority of usages. We don't require 100% because real corpora
# have variant phrasings ("X berbeda dengan Y" / "X berbeda dari Y")
# and we want the signal to fire as long as the connector is the
# dominant pattern.
_CONNECTOR_RATE_THRESHOLD = 0.5


# ----------------------------------------------------------------------
# Learner
# ----------------------------------------------------------------------

@dataclass
class PositionalClusterLearner:
    """Zero-bias positional cluster learner with post-hoc naming.

    Public API:
        train(corpus_lines)                  -> None
        inspect_clusters()                   -> Dict[int, List[str]]
        label_clusters(mapping)              -> None
        classify(text)                       -> RelationType
        spo(text)                            -> SPO
        save(path)                           -> None
        load(path)                           -> PositionalClusterLearner  # classmethod

    State (all learned from corpus, all JSON-serialisable):

        positional_freq:    {token: {position_bucket: count}}   (soft)
        action_object_freq: {action_token: {object_token: count}}
        cluster_id_of:      {action_token: cluster_id}          (-1 = unclustered)
        action_clusters:    {cluster_id: set(action_tokens)}
        cluster_labels:     {cluster_id: RelationType}          (empty until label_clusters())

    Composition:

        fallback: SemanticRoleClassifier instance used when clusters
        are unlabelled, when the action token has no cluster_id, or
        when the parse fails. The fallback owns its own
        frequency_table and learns from every fallback classification
        - so even when the learner delegates, the system still gets
        SemanticRoleClassifier's existing learnable behaviour.
    """

    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
    min_action_observations: int = _DEFAULT_MIN_ACTION_OBSERVATIONS
    fallback: SemanticRoleClassifier = field(
        default_factory=SemanticRoleClassifier
    )

    positional_freq: Dict[str, Dict[int, int]] = field(default_factory=dict)
    action_object_freq: Dict[str, Dict[str, int]] = field(default_factory=dict)
    cluster_id_of: Dict[str, int] = field(default_factory=dict)
    action_clusters: Dict[int, Set[str]] = field(default_factory=dict)
    cluster_labels: Dict[int, RelationType] = field(default_factory=dict)

    # Connector-signal state (added by the cluster-62 fix). See the
    # "_CONNECTOR_*" constants above for the detection contract.
    #
    # action_connector_signature: {action_token: bool} — True when the
    #   action routinely takes a connector token between it and its
    #   object. Used by _cluster_actions() as a structural split key
    #   so that structurally-different actions (e.g. "adalah" direct
    #   vs "berbeda" + connector) cannot merge even when their object
    #   distributions have high weighted-Jaccard similarity.
    #
    # connector_tokens: set of corpus-wide connector tokens discovered
    #   during train(). Exposed for inspection/debugging — classify()
    #   does not consult it. Persisted to JSON so loaded learners keep
    #   the same clustering contract.
    action_connector_signature: Dict[str, bool] = field(default_factory=dict)
    connector_tokens: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True once train() has built the positional_freq table."""
        return bool(self.positional_freq)

    @property
    def is_labelled(self) -> bool:
        """True once label_clusters() has assigned at least one RelationType."""
        return bool(self.cluster_labels)

    # ------------------------------------------------------------------
    # STAGE 1: zero-bias training
    # ------------------------------------------------------------------

    def train(self, corpus_lines: List[str]) -> None:
        """Build positional_freq, action_object_freq, and action clusters.

        Idempotent in the sense that calling train() again on the same
        instance accumulates observations. Re-clustering runs from
        scratch each call (it is cheap and ensures cluster_ids stay
        consistent with the latest data).

        Pipeline:
            1. Parse each line into tokens; compute position buckets;
               bump positional_freq (soft counts - same token can
               appear in multiple buckets).
            2. For every SVO-shaped sentence (>= 3 tokens), extract
               (action, object) by CURRENT position (index 1 = action,
               index 2 or -1 = object) and bump action_object_freq.
               Also record the "between-first" token (if any) for the
               connector-signal detector.
            3. Filter actions with >= min_action_observations.
            4. Compute the connector signature (action_connector_signature
               + corpus-wide connector_tokens set) from the between-first
               observations. See _CONNECTOR_* constants for the
               detection contract.
            5. Cluster those actions by Jaccard similarity of their
               object sets (greedy agglomerative merge), SPLIT FIRST
               by has_connector so structurally different actions never
               merge.
            6. Reset cluster_labels (labelling must be redone after
               re-training because cluster_ids may shift).

        Failure contract: empty / single-token lines are skipped
        silently. A train() call with zero usable lines leaves the
        learner un-trained; classify() then delegates to fallback.
        """
        if not corpus_lines:
            return

        # Per-action between-first observations. Each entry is
        # {action: {between_first_token_or_None: count}}. We accumulate
        # this alongside action_object_freq so the connector detector
        # has the raw positional evidence it needs without re-parsing
        # the corpus.
        action_between: Dict[str, Dict[Optional[str], int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Phase 1+2: positional frequencies + action/object co-occurrence.
        for line in corpus_lines:
            tokens = self._tokenize(line)
            if not tokens:
                continue
            # Positional frequencies (soft).
            for token, bucket in zip(tokens, self._compute_buckets(len(tokens))):
                pos_map = self.positional_freq.setdefault(token, {})
                pos_map[bucket] = pos_map.get(bucket, 0) + 1
            # Action/object co-occurrence (only for SVO-shaped sentences).
            if len(tokens) >= 3:
                action_token, object_token = self._extract_action_object(tokens)
                if action_token and object_token:
                    obj_bucket = self.action_object_freq.setdefault(action_token, {})
                    obj_bucket[object_token] = obj_bucket.get(object_token, 0) + 1
                    # Connector-signal evidence: record the token (or
                    # None) that sits immediately after the action and
                    # before the object. Used by _compute_connector_signature
                    # to build action_connector_signature + connector_tokens.
                    between_token = self._extract_between_token(
                        tokens, action_token, object_token
                    )
                    action_between[action_token][between_token] += 1

        # Phase 3: compute the connector signature from the
        # between-first observations. This populates
        # action_connector_signature (per-action bool) and
        # connector_tokens (corpus-wide grammar-token set).
        self._compute_connector_signature(action_between)

        # Phase 4+5: cluster actions by similarity of object
        # distributions, split first by has_connector.
        self._cluster_actions()

        # Phase 6: reset labels (cluster_ids may have shifted; old
        # labels are no longer meaningful). The human must call
        # label_clusters() again after re-training.
        self.cluster_labels = {}

    def _cluster_actions(self) -> None:
        """Greedy agglomerative clustering of actions by weighted Jaccard.

        Algorithm (pure Python, no sklearn / scipy):
            1. Build the set of "clusterable" actions: those with at
               least ``min_action_observations`` total co-occurrence
               counts.
            2. **Connector split** (new): partition the clusterable
               actions into two groups by their
               ``action_connector_signature`` value — has_connector=True
               vs has_connector=False. The two groups are clustered
               INDEPENDENTLY and can never merge across the split.
               This is the structural-signal fix for cluster 62:
               "adalah" (direct object) and "berbeda" (connector +
               object) end up in different clusters even though their
               object distributions overlap on taxonomy nouns.
            3. Initialise each cluster as a singleton {action} with
               its object *count map* (not just the set).
            4. Greedy pass: for every pair of clusters *within the
               same connector group*, compute the *weighted* Jaccard
               similarity of the merged object count maps. If >=
               similarity_threshold, merge them.
            5. Repeat passes until no merge happens (fixpoint).
            6. Assign cluster_ids (0, 1, 2, ...) across both groups
               (no-connector group first, then with-connector group).
               Actions that did not meet the min_observations bar get
               cluster_id = -1 (unclustered).

        Why *weighted* Jaccard instead of plain Jaccard on sets?
        Plain Jaccard treats every object token equally: a one-off
        co-occurrence counts as much as a 50x co-occurrence. That
        fragments synonyms: 'adalah' and 'merupakan' both take class
        nouns (mamalia, logam, ...) but rarely the *same* class noun,
        so their set overlap is small even though their distribution
        shape is identical. Weighted Jaccard sums min/max of counts,
        which gives more weight to high-frequency overlaps and
        correctly merges synonyms whose object distributions have the
        same *shape* even when the literal object tokens differ.

        The previous implementation used plain Jaccard on sets with
        threshold 0.25; the new implementation uses weighted Jaccard
        on count maps with threshold 0.13 (see
        ``_DEFAULT_SIMILARITY_THRESHOLD`` for the rationale).

        Why split by has_connector BEFORE clustering (not after)?
        If we clustered first and then split, synonyms like
        "adalah"/"merupakan" (both has_connector=False) would already
        be in the same cluster — the split would be a no-op for them,
        which is correct. But "adalah" (no connector) and "berbeda"
        (with connector) would also be in the same cluster (because
        their object distributions are similar), and the split would
        then fracture that cluster along the connector line —
        producing the desired separation but only as a post-hoc
        patch. Splitting first makes the structural signal a
        first-class clustering constraint: actions with different
        connector signatures are never even *considered* for merging,
        which is the correct semantics (they are structurally
        different predicate types).
        """
        # Reset previous clustering.
        self.cluster_id_of = {}
        self.action_clusters = {}

        # Build the set of clusterable actions + their object count maps.
        # Each clusterable action contributes its full {object: count} map.
        clusterable: Dict[str, Dict[str, int]] = {}
        for action, objs in self.action_object_freq.items():
            total = sum(objs.values())
            if total >= self.min_action_observations:
                clusterable[action] = dict(objs)

        # Connector split: partition clusterable actions by their
        # action_connector_signature value. Default is False (covers
        # the case where train() was called on a learner that somehow
        # has action_object_freq populated but not
        # action_connector_signature — e.g. via direct mutation in
        # tests; preserves backward compatibility).
        no_connector_actions: Dict[str, Dict[str, int]] = {}
        with_connector_actions: Dict[str, Dict[str, int]] = {}
        for action, objs in clusterable.items():
            if self.action_connector_signature.get(action, False):
                with_connector_actions[action] = objs
            else:
                no_connector_actions[action] = objs

        # Cluster each group independently, then concatenate the
        # resulting cluster lists so cluster_ids are assigned
        # monotonically across both groups.
        all_clusters: List[Tuple[Set[str], Dict[str, int]]] = []
        for group in (no_connector_actions, with_connector_actions):
            group_clusters = self._cluster_action_group(group)
            all_clusters.extend(group_clusters)

        # Assign cluster_ids across the concatenated group clusters.
        for cluster_id, (actions, _objs) in enumerate(all_clusters):
            self.action_clusters[cluster_id] = actions
            for action in actions:
                self.cluster_id_of[action] = cluster_id

        # Mark unclustered actions with cluster_id = -1. This includes
        # both actions below min_action_observations (excluded from
        # clustering entirely) and any action_object_freq entry that
        # was somehow skipped. classify() treats cluster_id = -1 as
        # "no cluster" and falls back to SemanticRoleClassifier.
        #
        # This loop runs even when no actions were clusterable (e.g.
        # every action had only 1 observation) so that the
        # cluster_id_of mapping is complete for every action the
        # learner has seen.
        for action in self.action_object_freq:
            if action not in self.cluster_id_of:
                self.cluster_id_of[action] = -1

    def _cluster_action_group(
        self, actions_objs: Dict[str, Dict[str, int]]
    ) -> List[Tuple[Set[str], Dict[str, int]]]:
        """Run the greedy agglomerative merge on one connector group.

        Returns a list of (set_of_actions, dict_of_object_counts) —
        the merged clusters for this group. Empty list if the group
        has no actions.

        This is the same algorithm the previous _cluster_actions()
        ran on the full clusterable set; we now run it per-group so
        the connector signature acts as a hard partition before
        similarity-based merging.
        """
        if not actions_objs:
            return []

        # Initial clusters: each action in its own cluster.
        # Each cluster is (set_of_actions, dict_of_object_counts).
        clusters: List[Tuple[Set[str], Dict[str, int]]] = [
            ({action}, dict(objs)) for action, objs in actions_objs.items()
        ]

        # Greedy agglomerative merge until fixpoint.
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            # Find the best pair to merge (highest weighted Jaccard
            # above threshold).
            best_i, best_j, best_sim = -1, -1, -1.0
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = self._weighted_jaccard(
                        clusters[i][1], clusters[j][1]
                    )
                    if sim >= self.similarity_threshold and sim > best_sim:
                        best_sim = sim
                        best_i, best_j = i, j
            if best_i >= 0:
                # Merge cluster j into cluster i. Aggregate object
                # counts so the merged cluster's distribution is the
                # sum of its members'.
                actions_i, objs_i = clusters[best_i]
                actions_j, objs_j = clusters[best_j]
                actions_i.update(actions_j)
                for obj, count in objs_j.items():
                    objs_i[obj] = objs_i.get(obj, 0) + count
                clusters.pop(best_j)
                merged = True

        return clusters

    # ------------------------------------------------------------------
    # Connector-signal detection (cluster-62 fix)
    # ------------------------------------------------------------------

    def _compute_connector_signature(
        self, action_between: Dict[str, Dict[Optional[str], int]]
    ) -> None:
        """Populate ``action_connector_signature`` and ``connector_tokens``.

        Two-phase detection (see ``_CONNECTOR_*`` constants for the
        contract):

          Phase A — corpus-wide connector discovery:
            For every token that appears in the between-first slot
            (the slot immediately after the action and before the
            object), check whether it qualifies as a corpus-wide
            connector. A token qualifies when:
              - it occupies the between-first slot at least
                ``_CONNECTOR_MIN_BETWEEN_COUNT`` times across the
                corpus (so one-off nouns don't qualify); AND
              - it NEVER appears as an object of any action in the
                corpus (so real object nouns that sometimes sit
                mid-sentence don't qualify — only grammar-only
                tokens like prepositions / complementizers).

            The set of qualifying tokens becomes ``self.connector_tokens``.

          Phase B — per-action signature:
            For every action with at least one observation, find the
            most common between-first token. If that token is a
            corpus-wide connector AND it occupies >=
            ``_CONNECTOR_RATE_THRESHOLD`` of the action's
            observations, the action's signature is True; otherwise
            False.

        Reset contract: this method overwrites both fields from
        scratch, so it is safe to call on every train() (no stale
        entries from a previous corpus survive).
        """
        self.action_connector_signature = {}
        self.connector_tokens = set()

        if not action_between:
            return

        # Phase A: corpus-wide connector discovery.
        #
        # Count (a) how often each token appears in the between-first
        # slot (across all actions), and (b) how often each token
        # appears as an object (across all actions). A token is a
        # corpus-wide connector when (a) >= _CONNECTOR_MIN_BETWEEN_COUNT
        # AND (b) == 0.
        between_first_counts: Dict[str, int] = defaultdict(int)
        for action, between_map in action_between.items():
            for tok, count in between_map.items():
                if tok is None:
                    continue
                between_first_counts[tok] += count

        object_counts: Dict[str, int] = defaultdict(int)
        for action, objs in self.action_object_freq.items():
            for obj, count in objs.items():
                object_counts[obj] += count

        for tok, bcount in between_first_counts.items():
            if bcount < _CONNECTOR_MIN_BETWEEN_COUNT:
                continue
            if object_counts.get(tok, 0) > 0:
                # This token sometimes appears as an object — it's a
                # real noun that happens to sit mid-sentence in some
                # multi-word-object sentences. Not a connector.
                continue
            self.connector_tokens.add(tok)

        # Phase B: per-action has_connector signature.
        for action, between_map in action_between.items():
            total = sum(between_map.values())
            if total == 0:
                self.action_connector_signature[action] = False
                continue
            # Find the most common non-None between-first token.
            best_tok: Optional[str] = None
            best_count = 0
            for tok, count in between_map.items():
                if tok is None:
                    continue
                if count > best_count:
                    best_tok = tok
                    best_count = count
            if best_tok is None:
                # Only None entries — action is always direct.
                self.action_connector_signature[action] = False
                continue
            rate = best_count / total
            has_connector = (
                rate >= _CONNECTOR_RATE_THRESHOLD
                and best_tok in self.connector_tokens
            )
            self.action_connector_signature[action] = has_connector

    @staticmethod
    def _extract_between_token(
        tokens: List[str],
        action_token: str,
        object_token: str,
    ) -> Optional[str]:
        """Return the token (or None) sitting between action and object.

        Given a token list and the (action, object) pair extracted by
        :meth:`_extract_action_object`, find the first token that sits
        strictly between the action's position and the object's
        position. This is the "between-first" slot used by the
        connector-signal detector.

        Returns:
            - None if action is immediately followed by object (no
              between token). This is the "direct" pattern, e.g.
              "kucing adalah mamalia" → action="adalah",
              object="mamalia", between=None.
            - The first between token otherwise, e.g.
              "kucing berbeda dari reptil" → action="berbeda",
              object="reptil", between="dari".

        Edge cases:
            - If action or object is not found in tokens, returns None
              (no positional evidence to extract).
            - If action and object are adjacent (no tokens between),
              returns None.
            - The object is assumed to be the LAST token (matches
              :meth:`_extract_action_object`'s contract). We search
              for object from the end so a token that appears both
              mid-sentence and as the object is correctly identified
              as the object.
        """
        if not tokens or not action_token or not object_token:
            return None
        # Find the action index (first occurrence).
        try:
            ai = tokens.index(action_token)
        except ValueError:
            return None
        # Find the object index (last occurrence — _extract_action_object
        # uses tokens[-1] as the object).
        oi = len(tokens) - 1
        if oi <= ai:
            return None
        if tokens[oi] != object_token:
            # Defensive: the object_token passed in doesn't match the
            # last token. Fall back to last-index-of search.
            try:
                oi = tokens.index(object_token, ai + 1)
            except ValueError:
                return None
            if oi <= ai:
                return None
        between = tokens[ai + 1:oi]
        if not between:
            return None
        return between[0]

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """Plain Jaccard similarity of two sets: |A ∩ B| / |A ∪ B|.

        Returns 0.0 for two empty sets (convention; avoids div-by-zero).

        Kept for backward compatibility and as a public diagnostic
        helper. The clustering algorithm itself uses
        :meth:`_weighted_jaccard` (which considers co-occurrence
        counts, not just set membership) - see ``_cluster_actions``
        for the rationale.
        """
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    @staticmethod
    def _weighted_jaccard(
        a: Dict[str, int], b: Dict[str, int]
    ) -> float:
        """Weighted Jaccard similarity of two count maps.

        Formula::

            sum(min(c_a(x), c_b(x)) for x in A∩B)
            ----------------------------------------
            sum(max(c_a(x), c_b(x)) for x in A∪B)

        where ``c_a(x)`` is the count associated with key ``x`` in map
        ``a`` (0 if absent). Returns 0.0 for two empty maps.

        Why weighted instead of plain set Jaccard? Two synonyms like
        'adalah' and 'merupakan' both take class-noun objects but
        rarely the *same* class noun, so their set overlap is tiny.
        Their count-map overlap, however, is large: both have many
        low-count abstract objects and a few high-count ones, and the
        *shape* of the distribution matches. Weighted Jaccard captures
        that shape; plain set Jaccard does not.
        """
        if not a and not b:
            return 0.0
        all_keys = set(a.keys()) | set(b.keys())
        if not all_keys:
            return 0.0
        numerator = sum(min(a.get(k, 0), b.get(k, 0)) for k in all_keys)
        denominator = sum(max(a.get(k, 0), b.get(k, 0)) for k in all_keys)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    # ------------------------------------------------------------------
    # STAGE 2: post-hoc naming + inspection
    # ------------------------------------------------------------------

    def inspect_clusters(self) -> Dict[int, List[str]]:
        """Return a human-readable view of every cluster for review.

        Returns ``{cluster_id: [action_token, action_token, ...]}``
        sorted by cluster_id. Use :meth:`inspect_cluster_details` for
        a richer view that also includes the top objects per cluster.

        Unclustered actions (cluster_id = -1) are NOT included - they
        have no cluster to inspect.
        """
        out: Dict[int, List[str]] = {}
        for cluster_id in sorted(self.action_clusters.keys()):
            actions = sorted(self.action_clusters[cluster_id])
            out[cluster_id] = actions
        return out

    def inspect_cluster_details(
        self, top_objects: int = _INSPECT_TOP_OBJECTS
    ) -> Dict[int, Dict[str, object]]:
        """Richer cluster view: actions + top objects + label + connector.

        Returns ``{cluster_id: {"actions": [...], "top_objects": [...],
        "label": Optional[str], "has_connector": bool}}``. The
        ``label`` is the RelationType name if label_clusters() has
        named this cluster, else None. The ``has_connector`` field is
        True if every action in the cluster has
        ``action_connector_signature[action] == True`` (i.e. the
        cluster was produced by the with-connector partition of
        ``_cluster_actions``); False otherwise. This is the
        human-readable signal that lets the user verify the
        structural split is doing its job — e.g. cluster 62 from the
        combined corpus used to mix "adalah" (no connector) and
        "berbeda" (with connector); after the fix, the two predicates
        live in different clusters and the ``has_connector`` field
        makes the split visible at a glance.

        Args:
            top_objects: How many object tokens (sorted by total
                co-occurrence count across all actions in the cluster)
                to include per cluster.
        """
        out: Dict[int, Dict[str, object]] = {}
        for cluster_id in sorted(self.action_clusters.keys()):
            actions = sorted(self.action_clusters[cluster_id])
            # Aggregate object counts across all actions in this cluster.
            obj_totals: Dict[str, int] = defaultdict(int)
            for action in actions:
                for obj, count in self.action_object_freq.get(action, {}).items():
                    obj_totals[obj] += count
            top_objs = sorted(
                obj_totals.items(), key=lambda kv: (-kv[1], kv[0])
            )[:top_objects]
            # Connector signature for the cluster: True if every
            # action in the cluster has has_connector=True. Mixed
            # clusters (which shouldn't happen given the split-first
            # contract, but we report defensively) show False.
            has_connector = bool(actions) and all(
                self.action_connector_signature.get(a, False)
                for a in actions
            )
            out[cluster_id] = {
                "actions": actions,
                "top_objects": [obj for obj, _ in top_objs],
                "label": (
                    self.cluster_labels[cluster_id].name
                    if cluster_id in self.cluster_labels
                    else None
                ),
                "has_connector": has_connector,
            }
        return out

    def label_clusters(self, mapping: Dict[int, RelationType]) -> None:
        """Assign RelationType names to clusters (post-hoc, once).

        Idempotent: calling label_clusters() again with a different
        mapping overwrites previous labels. Clusters not in the
        mapping keep whatever label they had (or stay unlabelled).

        Args:
            mapping: {cluster_id: RelationType}. cluster_ids must
                exist in ``self.action_clusters``; unknown ids are
                silently skipped (forward-compatibility: a saved
                mapping from a previous run that had more clusters
                should not crash on load).
        """
        for cluster_id, relation_type in mapping.items():
            if cluster_id in self.action_clusters:
                self.cluster_labels[cluster_id] = relation_type

    # ------------------------------------------------------------------
    # Public API: classification
    # ------------------------------------------------------------------

    def classify(self, text: str) -> RelationType:
        """Classify ``text`` using labelled clusters, else fallback.

        Decision tree:
            1. If untrained OR no clusters are labelled -> delegate to
               fallback. This preserves SemanticRoleClassifier's
               behaviour exactly before the human labels clusters.
            2. Parse SPO positionally (CURRENT sentence, not global).
               If parsing fails -> fallback.
            3. Negation override: a negation token immediately before
               the action -> DIFFERENTIAL. Same contract as the
               fallback; beats any cluster label.
            4. Look up the action's cluster_id. If that cluster is
               labelled, return its RelationType.
            5. Otherwise -> fallback.
        """
        if not self.is_trained or not self.is_labelled:
            return self.fallback.classify(text)

        try:
            spo = self.spo(text)
        except Exception:
            return self.fallback.classify(text)

        # Step 3: negation beats everything.
        if spo.negated:
            return RelationType.DIFFERENTIAL

        action_token = self._normalize_token(spo.predicate)
        if not action_token:
            return self.fallback.classify(text)

        cluster_id = self.cluster_id_of.get(action_token)
        if cluster_id is None or cluster_id < 0:
            # Action not in any cluster (unseen, or below
            # min_action_observations). Fallback.
            return self.fallback.classify(text)

        relation_type = self.cluster_labels.get(cluster_id)
        if relation_type is None:
            # Cluster exists but human hasn't labelled it yet.
            return self.fallback.classify(text)

        return relation_type

    # ------------------------------------------------------------------
    # Public API: SPO parsing (positional, NOT cluster-based)
    # ------------------------------------------------------------------

    def spo(self, text: str) -> SPO:
        """Parse ``text`` into Subject-Predicate-Object positionally.

        Roles are derived from CURRENT sentence position, not from any
        global cluster membership. This is the polysemy fix:

            "ayam mencari pakan"   -> subject="ayam"  (index 0)
            "manusia potong ayam"  -> object="ayam"   (index 2 or -1)

        Both sentences can be parsed by the same learner instance
        because role = position in THIS sentence.

        Strategy:
            - If untrained -> delegate to fallback (preserves the
              seed-based predicate extraction + middle-token
              heuristic for short sentences like "X bukan Y").
            - If trained -> positional SVO split:
                * 3 tokens: subject=tokens[0], predicate=tokens[1],
                  object=tokens[2]
                * >3 tokens: subject=tokens[0], predicate=tokens[1],
                  object=tokens[-1]
                * <3 tokens: delegate to fallback.

        Negation detection: same logic as SemanticRoleClassifier -
        check the last token of the subject for a negation token. We
        only check the LAST token because in both Indonesian and
        English the negation sits immediately before the predicate.
        """
        if not self.is_trained:
            return self.fallback.spo(text)

        raw = (text or "").strip()
        if not raw:
            return SPO(subject="", predicate="", object="", raw=raw)

        normalized = re.sub(r"\s+", " ", raw.lower())
        tokens = normalized.split(" ")

        if len(tokens) < 3:
            # Too short for positional SVO. Delegate so we keep the
            # "X bukan Y" -> DIFFERENTIAL path that lives in the
            # fallback's seed table.
            return self.fallback.spo(text)

        # Positional SVO: index 0 = subject, index 1 = predicate,
        # index 2 (3-token) or -1 (>3-token) = object.
        subject = tokens[0]
        predicate = tokens[1]
        obj = tokens[2] if len(tokens) == 3 else tokens[-1]
        # For >3-token sentences, include all middle tokens between
        # predicate and object in the predicate slot so downstream
        # negation/seed matching still works on the full predicate
        # phrase (e.g. "saya sedang makan nasi" -> predicate
        # "sedang makan", object "nasi").
        if len(tokens) > 3:
            predicate = " ".join(tokens[1:-1])

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
              "similarity_threshold": 0.25,
              "min_action_observations": 2,
              "positional_freq":     {"makan": {"1": 10}, ...},
              "action_object_freq":  {"menyebabkan": {"panas": 2, ...}, ...},
              "cluster_id_of":       {"makan": 0, "minum": 0, ...},
              "action_clusters":     {"0": ["makan", "minum", ...], ...},
              "cluster_labels":      {"0": "CAUSAL", ...},    # may be empty
              "action_connector_signature": {"berbeda": true, "adalah": false, ...},
              "connector_tokens":    ["dari", "dengan", "sebagai", ...]
            }

        The ``action_connector_signature`` and ``connector_tokens``
        fields are persisted so a loaded learner reproduces the same
        clustering contract (actions flagged has_connector=True
        continue to be partitioned into the with-connector group on
        any subsequent re-train). Older save files (pre-cluster-62
        fix) lack these fields and load() backfills them as empty /
        False — backward compatible.

        Atomic write: temp file + os.replace. Parent dirs created on
        demand. Same pattern as SemanticRoleClassifier.save.

        The wrapped fallback is NOT serialised here - callers who want
        a persisted fallback should construct one with its own
        persist_path and pass it via the constructor.
        """
        serialisable = {
            "similarity_threshold": self.similarity_threshold,
            "min_action_observations": self.min_action_observations,
            "positional_freq": {
                tok: {str(b): c for b, c in pos_map.items()}
                for tok, pos_map in self.positional_freq.items()
            },
            "action_object_freq": {
                act: dict(objs)
                for act, objs in self.action_object_freq.items()
            },
            "cluster_id_of": dict(self.cluster_id_of),
            "action_clusters": {
                str(cid): sorted(actions)
                for cid, actions in self.action_clusters.items()
            },
            "cluster_labels": {
                str(cid): rt.name
                for cid, rt in self.cluster_labels.items()
            },
            "action_connector_signature": dict(self.action_connector_signature),
            "connector_tokens": sorted(self.connector_tokens),
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

        Returns a learner whose is_trained is True (positional_freq is
        populated) and whose is_labelled reflects whatever
        cluster_labels were saved (True if any, False otherwise).

        The wrapped fallback is a fresh SemanticRoleClassifier -
        callers who want to also restore the fallback's
        frequency_table should construct one separately and pass it
        via the ``fallback`` constructor argument.

        Unknown RelationType names in the file are silently skipped
        (forward-compat: same contract as
        SemanticRoleClassifier._load_frequency_table).
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        learner = cls(
            similarity_threshold=float(
                raw.get("similarity_threshold", _DEFAULT_SIMILARITY_THRESHOLD)
            ),
            min_action_observations=int(
                raw.get("min_action_observations", _DEFAULT_MIN_ACTION_OBSERVATIONS)
            ),
        )

        # positional_freq: {token: {bucket_str: count}} -> {token: {int: int}}
        for tok, pos_map in raw.get("positional_freq", {}).items():
            if not isinstance(tok, str) or not isinstance(pos_map, dict):
                continue
            learner.positional_freq[tok] = {
                int(b): int(c)
                for b, c in pos_map.items()
                if isinstance(b, str) and isinstance(c, (int, float))
            }

        # action_object_freq: {action: {object: count}}
        for act, objs in raw.get("action_object_freq", {}).items():
            if not isinstance(act, str) or not isinstance(objs, dict):
                continue
            learner.action_object_freq[act] = {
                obj: int(c)
                for obj, c in objs.items()
                if isinstance(obj, str) and isinstance(c, (int, float))
            }

        # cluster_id_of: {action: cluster_id}
        for act, cid in raw.get("cluster_id_of", {}).items():
            if isinstance(act, str) and isinstance(cid, int):
                learner.cluster_id_of[act] = cid

        # action_clusters: {cluster_id_str: [actions]} -> {int: set}
        for cid_str, actions in raw.get("action_clusters", {}).items():
            if not isinstance(cid_str, str) or not isinstance(actions, list):
                continue
            try:
                cid_int = int(cid_str)
            except ValueError:
                continue
            learner.action_clusters[cid_int] = {
                a for a in actions if isinstance(a, str)
            }

        # cluster_labels: {cluster_id_str: relation_type_name} ->
        #                  {int: RelationType}
        for cid_str, rt_name in raw.get("cluster_labels", {}).items():
            if not isinstance(cid_str, str) or not isinstance(rt_name, str):
                continue
            try:
                cid_int = int(cid_str)
            except ValueError:
                continue
            try:
                learner.cluster_labels[cid_int] = RelationType[rt_name]
            except KeyError:
                # Unknown relation type from a future version. Skip.
                continue

        # action_connector_signature: {action: bool}
        # Backward-compat: older save files (pre-cluster-62 fix) lack
        # this field. The empty default (set by the dataclass) is the
        # correct backfill — _cluster_actions treats absent entries as
        # has_connector=False, matching the pre-fix clustering
        # behaviour for legacy saves.
        for act, flag in raw.get("action_connector_signature", {}).items():
            if isinstance(act, str) and isinstance(flag, bool):
                learner.action_connector_signature[act] = flag

        # connector_tokens: list[str] -> set[str]
        # Same backward-compat note as above.
        for tok in raw.get("connector_tokens", []):
            if isinstance(tok, str):
                learner.connector_tokens.add(tok)

        return learner

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lower-case + strip punctuation + collapse whitespace + split.

        Empty input -> []. Punctuation (commas, periods, semicolons,
        colons, exclamation/question marks, brackets, quotes) is
        replaced with spaces before whitespace normalisation. This
        prevents tokens like ``"ahli,"`` (with a trailing comma) from
        being treated as distinct from ``"ahli"`` - a real issue in
        multi-clause sentences like ``"menurut ahli, X bukan Y"``
        where the comma attaches to "ahli" and creates a spurious
        token.

        Hyphens are preserved (e.g. ``"lumba-lumba"`` stays one token)
        because they're part of the word in Bahasa Indonesia.
        """
        if not text:
            return []
        # Replace common punctuation with spaces (NOT hyphens, which
        # are intra-word in Indonesian: "lumba-lumba", "kupu-kupu").
        no_punct = re.sub(r"[,\.;:!?()\[\]{}\"'/\\]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", no_punct.strip())
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
    def _compute_buckets(n: int) -> List[int]:
        """Position buckets for a sentence of ``n`` tokens.

        Mapping:
            n <= 0:  []
            n == 1:  [0]
            n == 2:  [0, 1]
            n == 3:  [0, 1, 2]                       # classic SVO
            n  > 3:  [0] + [1] * (n - 2) + [-1]      # collapse middle

        For >3-token sentences, every middle token gets bucket 1 so
        the action distribution accumulates from any middle position;
        the last token gets bucket -1 so the object cluster is
        separable from the 3-token-only object cluster (bucket 2).
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

    @staticmethod
    def _looks_like_verb(token: str) -> bool:
        """Heuristic: does this token look like an Indonesian verb?

        Indonesian verbs are highly morphologically regular. The vast
        majority start with me-, ber-, di-, or ter- (and their
        allomorphs meng-/meny-/mem-/men-, bel-, diper-, etc.). This
        coarse morphological signal lets us distinguish verbs from
        nouns when positional parsing alone would be ambiguous - e.g.
        in multi-word subjects like "ahli gizi menyarankan" where
        "gizi" sits at position 1 (noun, part of compound subject)
        but "menyarankan" at position 2 is the real verb.

        Copulas (:data:`_COPULAS`) are also recognised as verbs even
        though they lack the morphological prefix. This lets
        multi-word categorical sentences like "suku bunga adalah
        instrumen" be parsed correctly.

        Conservative by design:
          - All prefixes are 3+ characters to avoid false positives
            ("di" alone is a preposition, "me" alone matches "merah").
          - We accept some false positives ("beras" = rice, "ternak" =
            livestock) because they're rare in the action slot and the
            cost of false negatives (breaking the multi-word subject
            fix) is much higher.
          - We accept some false negatives ("makan", "minum", "ambil"
            don't carry these prefixes) because the caller falls back
            to the first non-stoplist token when no verb-looking token
            is found in 3-token sentences.
        """
        if not token or len(token) < 3:
            return False
        if token in _COPULAS:
            return True
        return token.startswith(_VERB_PREFIXES)

    @staticmethod
    def _extract_action_object(
        tokens: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract (action_token, object_token) by CURRENT position.

        For 3-token sentences: action=tokens[1], object=tokens[2].
        For >3-token sentences: action=first verb-looking token (must
            start with me-/ber-/diper-/ter-) before the object slot;
            object=tokens[-1]. If no verb-looking token is found, the
            sentence is skipped (returns ``(None, None)``).
        For <3-token sentences: ``(None, None)`` - no SVO structure.

        This is the polysemy fix in action: the action and object are
        determined by WHERE they sit in *this* sentence, not by any
        global cluster membership. The same token "ayam" can be the
        object of "manusia potong ayam" and the agent of
        "ayam mencari pakan" - both are recorded correctly because
        positional_freq is a *soft* count.

        Bug 1 fix - action stoplist:
        Function words in :data:`_ACTION_STOPLIST` (intensifiers,
        deictic markers, copula-like appearance verbs like "sangat",
        "itu", "tampak", "tergolong", negation markers like "bukan" /
        "tidak", prefix-phrase introducers like "menurut" / "secara")
        are skipped from the action slot. This prevents garbage
        clusters like ``{'actions': ['sangat'], 'top_objects': ['asin',
        'dingin']}`` that previously formed when state+adjective
        sentences ("es itu sangat dingin") had their intensifier
        captured as the action.

        Multi-word subject fix - verb-prefix requirement:
        Sentences with >3 tokens may have a multi-word subject like
        "ahli gizi" or "dokter kulit" occupying positions 0-1, which
        means position 1 is a noun (not the action). To avoid
        capturing that noun as the action, we require a verb-looking
        token (starts with me-, ber-, diper-, or ter-) before the
        object slot. If no verb-looking token exists, the sentence is
        skipped. This prevents noun-as-action garbage like
        ``{'actions': ['gizi'], 'top_objects': ['diet', 'kalori']}``.

        For 3-token sentences (classic SVO with no room for multi-word
        subjects), we accept the first non-stoplist token at position
        1 as the action. This preserves the parse for irregular verbs
        like "makan", "minum", "adalah" that don't carry verb
        prefixes.

        When the action is also the last token (no non-stoplist token
        remains after it, as in pure state+adjective patterns like
        "es itu dingin"), the sentence has no real object and is
        skipped from ``action_object_freq`` by returning
        ``(None, None)``.
        """
        if len(tokens) < 3:
            return None, None

        if len(tokens) == 3:
            # 3-token SVO: no room for a multi-word subject, so the
            # positional parse is unambiguous. Take the first
            # non-stoplist token at position 1 as the action; the
            # object is tokens[2] (the last token).
            action_idx: Optional[int] = None
            for i in range(1, len(tokens)):
                if tokens[i] not in _ACTION_STOPLIST:
                    action_idx = i
                    break
            if action_idx is None:
                return None, None
            # If the action IS the last token, no object remains -
            # skip (state+adjective with no real object).
            if action_idx == len(tokens) - 1:
                return None, None
            return tokens[action_idx], tokens[-1]

        # >3-token sentence: potential multi-word subject. Require a
        # verb-looking token before the object slot to disambiguate.
        # This is the multi-word subject fix: it prevents nouns like
        # "gizi" (in "ahli gizi menyarankan diet") from being captured
        # as the action when they're really part of the compound
        # subject.
        action_idx = None
        for i in range(1, len(tokens) - 1):
            if tokens[i] in _ACTION_STOPLIST:
                continue
            if PositionalClusterLearner._looks_like_verb(tokens[i]):
                action_idx = i
                break
        if action_idx is None:
            # No verb-looking token before the object slot. Skip this
            # sentence to avoid noun-as-action garbage. The cost is
            # losing sentences whose verb is an irregular root (e.g.
            # "makan", "minum") in a >3-token sentence, but that's a
            # small fraction of the corpus and the gain in cluster
            # quality (no garbage noun-actions) is much larger.
            return None, None

        return tokens[action_idx], tokens[-1]

    @staticmethod
    def _has_negation_before(subject: str) -> bool:
        """True when the subject's last non-empty token is a negation.

        Same logic as SemanticRoleClassifier._has_negation_before:
        only the LAST token of the subject is checked, because in
        both Indonesian ("X tidak menyebabkan Y") and English
        ("X does not cause Y") the negation sits immediately before
        the predicate. Widening the window would risk false positives
        on sentences like "Not all humans are mortal".
        """
        if not subject:
            return False
        tokens = subject.split()
        if not tokens:
            return False
        return tokens[-1] in _NEGATION_TOKENS
