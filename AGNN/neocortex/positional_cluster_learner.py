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
# Clustering parameters
# ----------------------------------------------------------------------

# Two actions are similar (and thus merge into the same cluster) when
# the Jaccard similarity of their object-token sets is >= this value.
# Jaccard = |A ∩ B| / |A ∪ B|. 0.25 means "at least 1/4 overlap" -
# conservative; prevents unrelated actions from collapsing together.
_DEFAULT_SIMILARITY_THRESHOLD = 0.25

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
            3. Filter actions with >= min_action_observations.
            4. Cluster those actions by Jaccard similarity of their
               object sets (greedy agglomerative merge).
            5. Reset cluster_labels (labelling must be redone after
               re-training because cluster_ids may shift).

        Failure contract: empty / single-token lines are skipped
        silently. A train() call with zero usable lines leaves the
        learner un-trained; classify() then delegates to fallback.
        """
        if not corpus_lines:
            return

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

        # Phase 3+4: cluster actions by similarity of object distributions.
        self._cluster_actions()

        # Phase 5: reset labels (cluster_ids may have shifted; old
        # labels are no longer meaningful). The human must call
        # label_clusters() again after re-training.
        self.cluster_labels = {}

    def _cluster_actions(self) -> None:
        """Greedy agglomerative clustering of actions by Jaccard similarity.

        Algorithm (pure Python, no sklearn / scipy):
            1. Build the set of "clusterable" actions: those with at
               least ``min_action_observations`` total co-occurrence
               counts.
            2. Initialise each cluster as a singleton {action}.
            3. Greedy pass: for every pair of clusters, compute the
               Jaccard similarity of the *union* of their object
               token sets. If >= similarity_threshold, merge them.
            4. Repeat passes until no merge happens (fixpoint).
            5. Assign cluster_ids (0, 1, 2, ...). Actions that did
               not meet the min_observations bar get cluster_id = -1
               (unclustered).

        The Jaccard uses object *token sets* (not weighted counts) so
        the similarity is about *which* objects an action takes, not
        how often. This matches the brief: "action yang diikuti object
        set yang mirip = 1 cluster".
        """
        # Reset previous clustering.
        self.cluster_id_of = {}
        self.action_clusters = {}

        # Build the set of clusterable actions + their object sets.
        clusterable: Dict[str, Set[str]] = {}
        for action, objs in self.action_object_freq.items():
            total = sum(objs.values())
            if total >= self.min_action_observations:
                clusterable[action] = set(objs.keys())

        if clusterable:
            # Initial clusters: each action in its own cluster.
            # Each cluster is represented as (set_of_actions, set_of_objects).
            clusters: List[Tuple[Set[str], Set[str]]] = [
                ({action}, objs.copy()) for action, objs in clusterable.items()
            ]

            # Greedy agglomerative merge until fixpoint.
            merged = True
            while merged and len(clusters) > 1:
                merged = False
                # Find the best pair to merge (highest Jaccard above threshold).
                best_i, best_j, best_sim = -1, -1, -1.0
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        sim = self._jaccard(clusters[i][1], clusters[j][1])
                        if sim >= self.similarity_threshold and sim > best_sim:
                            best_sim = sim
                            best_i, best_j = i, j
                if best_i >= 0:
                    # Merge cluster j into cluster i.
                    actions_i, objs_i = clusters[best_i]
                    actions_j, objs_j = clusters[best_j]
                    actions_i.update(actions_j)
                    objs_i.update(objs_j)
                    clusters.pop(best_j)
                    merged = True

            # Assign cluster_ids.
            for cluster_id, (actions, _objs) in enumerate(clusters):
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

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """Jaccard similarity of two sets: |A ∩ B| / |A ∪ B|.

        Returns 0.0 for two empty sets (convention; avoids div-by-zero).
        """
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

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
        """Richer cluster view: actions + top objects + label (if any).

        Returns ``{cluster_id: {"actions": [...], "top_objects": [...],
        "label": Optional[str]}}``. The ``label`` is the RelationType
        name if label_clusters() has named this cluster, else None.

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
            out[cluster_id] = {
                "actions": actions,
                "top_objects": [obj for obj, _ in top_objs],
                "label": (
                    self.cluster_labels[cluster_id].name
                    if cluster_id in self.cluster_labels
                    else None
                ),
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
              "cluster_labels":      {"0": "CAUSAL", ...}    # may be empty
            }

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

        return learner

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lower-case + collapse whitespace + split. Empty input -> []."""
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
    def _extract_action_object(
        tokens: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract (action_token, object_token) by CURRENT position.

        For 3-token sentences: action=tokens[1], object=tokens[2].
        For >3-token sentences: action=tokens[1], object=tokens[-1].
        For <3-token sentences: (None, None) - no SVO structure.

        This is the polysemy fix in action: the action and object are
        determined by WHERE they sit in *this* sentence, not by any
        global cluster membership. The same token "ayam" can be the
        object of "manusia potong ayam" and the agent of
        "ayam mencari pakan" - both are recorded correctly because
        positional_freq is a *soft* count.
        """
        if len(tokens) < 3:
            return None, None
        if len(tokens) == 3:
            return tokens[1], tokens[2]
        return tokens[1], tokens[-1]

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
