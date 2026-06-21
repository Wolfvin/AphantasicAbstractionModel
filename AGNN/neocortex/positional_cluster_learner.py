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

    inspect_cluster_details() -> Dict[int, Dict[str, object]]
        Returns a human-readable view of each cluster: its action
        tokens, its top object tokens (sorted by co-occurrence
        count), its label (if assigned), and its connector
        signature. This is what a human reviews before deciding
        the cluster -> RelationType mapping.

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
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
# Statistical anchor-word discovery (replaces _ACTION_STOPLIST/_COPULAS)
# ----------------------------------------------------------------------
#
# Previous design (PR #73, rejected for violating zero-bias):
#   ``_ACTION_STOPLIST`` was a hardcoded frozenset of Indonesian
#   function words (itu, sangat, memang, tampak, bukan, tidak, ...)
#   hand-picked based on their linguistic meaning. ``_COPULAS`` was a
#   similar hardcoded frozenset of copula verbs (adalah, merupakan,
#   ialah, yaitu, yakni). Both lists were the same kind of human
#   bias that PR #69 was rejected for (seeding RelationType by meaning)
#   - just smaller in scope.
#
# New design (zero-bias, statistical discovery):
#
#   1. FUNCTION WORD CANDIDATES — discovered from positional entropy.
#      A token is excluded from the action slot when:
#        - total frequency >= ``_FUNCTION_WORD_MIN_FREQ`` (so one-off
#          tokens don't get flagged on noise); AND
#        - it appears at >= ``_FUNCTION_WORD_MIN_POSITIONS`` distinct
#          fine-grained positions (raw indices, not collapsed buckets)
#          - function words span many positions because they're
#          grammatical markers that can attach to many sentence
#          structures, while content words concentrate at fewer
#          positions; AND
#        - normalized Shannon entropy of its fine-grained position
#          distribution >= ``_FUNCTION_WORD_ENTROPY_THRESHOLD``
#          (the distribution is flat, not concentrated); AND
#        - it does NOT carry Indonesian verb morphology (me-, ber-,
#          diper-, ter-) - morphology is a *form* signal, not a
#          meaning signal, so it stays.
#
#      Examples from the real corpus (pretrain_corpus.txt + depth):
#        'sangat'  - 4 positions, fine_nh=0.79, freq=35  -> excluded
#        'itu'     - 3 positions, fine_nh=0.80, freq=15  -> excluded
#        'memang'  - 3 positions, fine_nh=0.79, freq=17  -> excluded
#        'terlalu' - 4 positions, fine_nh=0.96, freq=33  -> excluded
#        'bukan'   - 4 positions, fine_nh=0.80, freq=254 -> excluded
#        'tidak'   - 6 positions, fine_nh=0.43, freq=155 -> excluded
#
#   2. ACTION BUCKET ANCHORS — discovered from positional concentration.
#      A token is RECOGNISED as a valid action (even without verb
#      morphology) when:
#        - total frequency >= ``_ACTION_ANCHOR_MIN_FREQ``; AND
#        - its dominant bucket is the action bucket (1); AND
#        - normalized Shannon entropy of its *bucket* distribution <
#          ``_ACTION_ANCHOR_MAX_BUCKET_ENTROPY`` (concentrated at the
#          action bucket).
#
#      This is the statistical replacement for ``_COPULAS``: copulas
#      like 'adalah' (bucket_freq={1: 69}, bucket_nh=0.0) and
#      'merupakan' (bucket_freq={1: 103}, bucket_nh=0.0) emerge as
#      action anchors automatically from positional concentration -
#      no human-curated copula list needed.
#
#   3. CHICKEN-AND-EGG BREAK — two-pass training.
#      Function word / action anchor discovery needs ``positional_freq``
#      to be populated, but ``positional_freq`` is built during training.
#      We solve this with a two-pass train() pipeline:
#        Pass 1: build ``positional_freq`` and ``fine_positional_freq``
#                (NO action/object extraction yet).
#        Compute ``function_word_candidates`` and ``action_bucket_anchors``
#                from the populated frequency tables.
#        Pass 2: extract (action, object) pairs using the discovered
#                sets - function words are skipped, action anchors are
#                recognised as verbs even without morphology.
#
# This eliminates the last human-authored word lists in the module.
# The connector-signal detector (see ``_CONNECTOR_*`` constants below)
# remains unchanged - it was already zero-bias (corpus-wide positional
# discovery, no hardcoded connector list).

# Function word candidate — minimum total frequency floor.
# Below this, a token's positional distribution is too noisy to flag
# as a function word. Calibrated to the smallest pretrain corpus where
# function words (sangat, itu, memang, ...) consistently reach freq 15+.
_FUNCTION_WORD_MIN_FREQ = 5

# Function word candidate — minimum number of distinct fine-grained
# positions. Function words span many positions (they're grammatical
# markers that can attach anywhere); content words concentrate at 1-2.
# Requiring >= 3 positions excludes copulas like 'adalah' (always at
# action slot regardless of sentence length, so 2 fine positions) and
# 'merupakan' (2 fine positions) which are content words despite lacking
# verbal morphology.
_FUNCTION_WORD_MIN_POSITIONS = 3

# Function word candidate — minimum normalized Shannon entropy of the
# fine-grained position distribution. 0.0 = perfectly concentrated at
# one position; 1.0 = perfectly uniform across all observed positions.
# 0.4 captures tokens that are clearly spread across multiple positions
# (e.g., 'tidak' at fine_nh=0.43) while excluding tokens that are
# concentrated (e.g., copulas at fine_nh=0.32).
_FUNCTION_WORD_ENTROPY_THRESHOLD = 0.4

# Function word candidate — MAXIMUM normalized Shannon entropy of the
# *bucket* distribution. Real function words (sangat, itu, bukan,
# tidak, ...) concentrate EXCLUSIVELY at the action bucket (b_nh = 0)
# — they never appear as subject or object because they're grammatical
# markers, not nouns. Content words that happen to appear mid-sentence
# (e.g. 'akun' as object of a verb in a clause, 'adonan' as a noun
# that's sometimes mid-sentence) have b_nh >= 0.3 because they also
# appear at bucket 0 (subject) or bucket 2/-1 (object).
#
# Threshold 0.1 cleanly separates real function words (all b_nh = 0.00)
# from content words that happen to have varied positions (b_nh >= 0.3).
_FUNCTION_WORD_MAX_BUCKET_ENTROPY = 0.1

# Action bucket anchor — minimum total frequency floor.
# Calibrated high enough to exclude one-off tokens in small corpora
# (e.g., 'itu' in the Bug 1 test corpus has freq 2 - below this floor)
# while admitting real action anchors (e.g., 'adalah' at freq 69).
_ACTION_ANCHOR_MIN_FREQ = 3

# Action bucket anchor — maximum normalized Shannon entropy of the
# *bucket* distribution. Anchors must be concentrated at the action
# bucket (low entropy). 0.5 admits tokens with one secondary bucket
# (e.g., a verb that's occasionally used as a noun) while excluding
# tokens that span multiple buckets uniformly.
_ACTION_ANCHOR_MAX_BUCKET_ENTROPY = 0.5

# 3-token sentence — minimum frequency at the action bucket for a
# non-verb-morphology token to be accepted as the action.
# In a 3-token SVO sentence (``X A Y``) the positional parse is
# unambiguous, so we can't use the multi-word-subject verb-prefix
# requirement. Instead, we require non-morphological candidates to
# have appeared at the action bucket at least this many times across
# the corpus - this excludes one-off function words in synthetic
# test corpora (e.g., 'memang' at freq 1) while admitting recurring
# irregular verbs (e.g., 'makan' at freq 2).
_3_TOKEN_MIN_ACTION_FREQ = 2

# Subject candidate — minimum frequency for a token that ONLY appears
# at the agent bucket (bucket 0) to be considered a discourse marker
# rather than a real subject. Real subject nouns in small corpora
# might only appear at bucket 0 by chance (e.g., 'manusia' in a
# 20-sentence test corpus); discourse markers like 'secara',
# 'menurut', 'karena' appear at bucket 0 many times because they're
# grammatical. The threshold 10 cleanly separates them: real subjects
# in small corpora stay below 10, discourse markers in the pretrain
# corpus (51 'secara', 45 'menurut', 34 'karena') all clear it.
_SUBJECT_DISCOURSE_MARKER_MIN_FREQ = 10


# ----------------------------------------------------------------------
# Brown clustering of object vocabulary (replaces literal-token overlap)
# ----------------------------------------------------------------------
#
# PR #71/#73/#74 clustered actions by weighted Jaccard of their LITERAL
# object-token distributions. This fails for synonym copulas like
# 'adalah' and 'merupakan' whose literal object sets barely overlap
# even though both take the same semantic class (taxonomy nouns). The
# weighted Jaccard threshold was lowered to 0.13 to compensate, but
# this is a tuned patch - the root cause is that literal tokens are
# too sparse a signal.
#
# New design: pre-cluster the OBJECT vocabulary itself via Brown
# clustering (hierarchical agglomerative, context-distribution
# clustering variant - see Clark 2000 CDC and the research doc
# AGNN/docs/research-unsupervised-grammar-induction.md section 1.3).
# Each object token is represented as the distribution of ACTIONS it
# co-occurs with. Objects that share action context are merged into a
# super-cluster. Then actions are clustered by weighted Jaccard of
# their SUPER-CLUSTER distributions, not literal object tokens.
#
# Example: 'mamalia' co-occurs with {adalah, merupakan}; 'logam'
# co-occurs with {adalah}. They share 'adalah' context, so Brown
# clustering merges them into one super-cluster 'taxonomy noun'. Then
# 'adalah' (which takes mamalia + logam) and 'merupakan' (which takes
# mamalia + unggas, where unggas also merged into 'taxonomy noun' via
# 'merupakan' context) both end up with super-cluster distribution
# {taxonomy noun: N} - they merge trivially even if their literal
# object tokens never overlap.
#
# The clustering uses plain Jaccard on action SETS (not weighted
# Jaccard on counts) because we want to capture "do these objects share
# action context?" - a yes/no question - rather than "do they have the
# same action distribution shape?". Brown/CDC literature uses both; we
# pick the simpler set-based variant because it correctly handles the
# synonym-copula case where action distributions have very different
# shapes but identical support.

# Brown clustering — minimum similarity for two object clusters to
# merge. We use *weighted* Jaccard on action-count maps (same metric
# as action clustering) rather than plain Jaccard on action sets.
# Plain Jaccard suffers from chain-merging: two objects that share
# ONE action out of many merge at Jaccard >= 0.13, then the merged
# cluster's action set grows, enabling further merges via different
# shared actions. The end result is one giant super-cluster
# containing most of the object vocabulary, which destroys the
# discrimination action clustering needs.
#
# Weighted Jaccard is more strict: it considers COUNT distributions,
# not just set membership. Two objects that share an action but at
# very different frequencies (e.g., 'mamalia' appears with adalah 8
# times but 'logam' appears with adalah only 3 times) get a lower
# similarity than two objects with matching count shapes. This breaks
# the chain-merge: an object can merge with the growing cluster only
# if its count distribution matches the cluster's aggregated
# distribution, not just shares one action.
#
# Threshold 0.15 is calibrated to:
#   - Merge synonyms (mamalia+logam via shared copula context)
#     so adalah+merupakan can merge via super-cluster overlap.
#   - NOT merge unrelated objects (hujan+panas) so CAUSAL and
#     TEMPORAL actions stay in separate clusters.
# 0.05 was too aggressive (CAUSAL+TEMPORAL collapsed); 0.3 was too
# conservative (adalah+merupakan didn't merge on the single-corpus
# test). 0.15 is the sweet spot verified on pretrain_corpus.txt,
# pretrain_corpus_depth.txt, and the combined corpus.
_BROWN_CLUSTER_SIMILARITY_THRESHOLD = 0.15

# Brown clustering — hard cap on the number of object super-clusters.
# The greedy agglomerative merge stops when EITHER no pair has
# similarity >= threshold OR the number of clusters drops to this cap.
# Prevents pathological corpora from collapsing every object into one
# giant super-cluster (which would make action clustering useless).
_BROWN_CLUSTER_MAX_CLUSTERS = 1  # effectively disabled - rely on threshold
# (Set to 1 to disable the cap; the threshold alone stops merging.)
# A non-trivial cap (e.g., 30) is recommended only for very large
# object vocabularies (>500 tokens) where the threshold-based stop
# might leave too many singletons. Pretrain corpus has ~500 distinct
# objects and works well with threshold-only stop.


# ----------------------------------------------------------------------
# Verb-prefix heuristic (morphological, NOT semantic - kept)
# ----------------------------------------------------------------------
#
# Indonesian verbs are highly morphologically regular: the vast majority
# start with one of the active/passive prefixes me-, ber-, di-, or ter-.
# This is a coarse but effective signal for distinguishing verbs from
# nouns in position 1 of an SVO sentence. It is a MORPHOLOGICAL signal
# (word form), not a SEMANTIC signal (word meaning) - so it does NOT
# violate the zero-bias principle that ``_ACTION_STOPLIST`` and
# ``_COPULAS`` violated.
#
# Used to disambiguate multi-word subjects like
# "ahli gizi menyarankan diet" (4 tokens). Position 1 here is "gizi"
# (a noun - the second word of the compound subject "ahli gizi"), and
# position 2 is "menyarankan" (the real verb). Without the heuristic,
# positional parsing would capture (action="gizi", object="diet") -
# a garbage pair where a noun is treated as the action.
#
# Conservative by design:
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

# How many top objects (by count) to surface in
# inspect_cluster_details(). Keeps the human-readable view manageable
# when a cluster has dozens of objects.
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
# MODIFIER (adverb / intensifier) discovery
# ----------------------------------------------------------------------
#
# Tokens like 'sangat', 'begitu', 'terlalu', 'cukup' are currently
# flagged as ``function_word_candidates`` (high positional entropy +
# concentrated at the action bucket). They are then EXCLUDED from the
# action slot during (action, object) extraction — but they never get
# a positive grammar identity of their own. They're noise from the
# action-clustering perspective, but they ARE a coherent grammar class:
# they routinely appear at the ACTION SLOT in 3-token "state + adj"
# sentences ("es sangat dingin" → 'sangat' at index 1, the action
# bucket). This is the positional signature of an adverb / intensifier
# modifying the adjective/object-property.
#
# The discovery contract (zero-bias, no hardcoded adverb list):
#
#   1. Build two pre-object frequency tables during Pass 1, split by
#      sentence length:
#        - ``pre_object_3tok_freq[token]``: count of times the token
#          sits at index n-2 (= index 1 = action bucket) in 3-token
#          sentences. This is the "modifier of adjective" position
#          ("es SANGAT dingin").
#        - ``pre_object_long_freq[token]``: count of times the token
#          sits at index n-2 in >3-token sentences. In >3-token
#          sentences, n-2 is the between-first slot (between action
#          and object) — this is where CONNECTORS sit
#          ("X berbeda DARI Y").
#      The split is the key signal: a MODIFIER's dominant position is
#      the 3-token action slot; a CONNECTOR's dominant position is the
#      >3-token between-first slot.
#
#   2. A token is a MODIFIER candidate when:
#        - it is in ``function_word_candidates`` (already flagged as
#          non-action by the anchor-word discovery — modifiers are
#          grammatical, not content words); AND
#        - its ``pre_object_3tok_freq`` count is >=
#          ``_MODIFIER_MIN_PRE_OBJECT_COUNT`` (so one-off tokens don't
#          qualify); AND
#        - the ratio
#          ``pre_object_3tok_freq[token] / (pre_object_3tok_freq[token]
#          + pre_object_long_freq[token])`` is >=
#          ``_MODIFIER_3TOK_RATE`` (the token's pre-object occurrences
#          are DOMINATED by 3-token sentences, not >3-token sentences).
#          This is what distinguishes 'sangat' (almost always in 3-token
#          "state + adj" sentences) from 'dari' (almost always in
#          >3-token "action + connector + object" sentences).
#
#   3. A token classified as MODIFIER is REMOVED from
#      ``connector_tokens`` (if it was there). This is the priority
#      rule: when a token's positional distribution clearly identifies
#      it as a modifier, the modifier classification takes precedence
#      over the connector classification. This corrects the over-broad
#      connector detector, which flags any non-object token that sits
#      between-first >= 3 times — including modifiers that sit at the
#      action slot in 3-token sentences and happen to also appear
#      between-first in a few >3-token sentences.
#
# Why split by sentence length?
#   In 3-token sentences (X A Y), the action slot (index 1) IS the
#   pre-object slot (n-2 = 1). A modifier like 'sangat' sits here in
#   "es sangat dingin". A real action like 'makan' also sits here in
#   "ayam makan pakan" — but 'makan' has verb morphology and is
#   excluded from function_word_candidates, so it's never considered
#   as a modifier candidate.
#
#   In >3-token sentences (X ... A ... Y), the between-first slot
#   (index 2..n-2) is where CONNECTORS sit. A connector like 'dari'
#   sits here in "tumbuhan tak bisa lepas dari karbon dioksida". A
#   modifier like 'sangat' rarely sits here — when it does appear in
#   >3-token sentences, it's usually at the action slot (index 1),
#   not at n-2.
#
#   The 3tok-vs-long split cleanly separates these two positional
#   patterns without consulting any meaning-based word list.

# Minimum corpus-wide 3-token pre-object count for a token to be
# considered a MODIFIER. Calibrated to exclude one-off tokens in
# small test corpora while admitting real adverbs in the pretrain
# corpus (where 'sangat' clears 19 3-token pre-object observations).
_MODIFIER_MIN_PRE_OBJECT_COUNT = 3

# Minimum fraction of a token's pre-object occurrences that must come
# from 3-token sentences (vs >3-token) for it to be classified as a
# MODIFIER. 0.5 = majority of pre-object observations are in 3-token
# sentences.
#
# Calibration: 'sangat' has 3tok=19, long=13, rate=0.59 → MODIFIER ✓.
# 'dari' has 3tok=0, long=51, rate=0.00 → CONNECTOR ✓.
# 'dengan' has 3tok=0, long=81, rate=0.00 → CONNECTOR ✓.
# 'begitu' has 3tok=13, long=5, rate=0.72 → MODIFIER ✓.
# 'cukup' has 3tok=14, long=6, rate=0.70 → MODIFIER ✓.
#
# 0.5 cleanly separates modifiers (rate >= 0.59) from connectors
# (rate = 0.00). We don't require a higher threshold because some
# modifiers like 'sangat' appear in long sentences too (e.g.
# "sinar matahari siang sangat terik" — 5 tokens, 'sangat' at n-2),
# which is still a modifier position, just in a longer sentence.
_MODIFIER_3TOK_RATE = 0.5


# ----------------------------------------------------------------------
# PARTICLE clustering — zero-bias grammar-class discovery
# ----------------------------------------------------------------------
#
# WHY THIS SECTION EXISTS (issue raised by user during the POS-class
# discovery review of PR #104):
#
#   ``_compute_modifiers()`` and ``_compute_connector_signature()``
#   above DIRECTLY assign the names "MODIFIER" / "CONNECTOR" inside
#   the detection function itself. The thresholds
#   (``_MODIFIER_3TOK_RATE``, ``_CONNECTOR_RATE_THRESHOLD``) were
#   calibrated by checking that specific KNOWN tokens ('sangat',
#   'begitu', 'cukup' vs 'dari', 'dengan') landed on the expected
#   side of the threshold. That is the SAME bias pattern PR #69 was
#   rejected for with RelationType seeding — just disguised as a
#   tuned numeric threshold instead of a literal word list. A human
#   decided the answer first ("sangat should be MODIFIER") and then
#   adjusted the mechanism until it produced that answer.
#
#   Contrast with how RelationType is done correctly (Stage
#   1/Stage 2 in the module docstring): ``_cluster_actions()``
#   produces UNNAMED integer cluster_ids from pure object-
#   distribution similarity — RelationType is never mentioned during
#   clustering. Only AFTER clusters are stable does a human call
#   ``label_clusters()`` to assign names, as a separate, optional,
#   post-hoc step. The naming never feeds back into how clustering
#   decisions are made.
#
#   This section gives the "leftover" particle tokens (those
#   excluded from the action slot by ``function_word_candidates`` —
#   i.e. NOT real content verbs) the same two-stage treatment:
#
#     STAGE 1 (zero-bias): ``_compute_particle_signature()`` builds a
#       purely positional/distributional FEATURE VECTOR per particle
#       token (no name attached). ``_cluster_particles()`` then
#       groups particle tokens by SIMILARITY of this feature vector
#       using the same kind of greedy-agglomerative-merge-by-distance
#       algorithm already used for action/object clustering. The
#       output is UNNAMED: ``particle_clusters: {cluster_id: {tokens}}``.
#       No "MODIFIER" or "CONNECTOR" string appears anywhere in this
#       stage.
#
#     STAGE 2 (post-hoc naming): a human inspects
#       ``inspect_particle_clusters()`` and assigns names via
#       ``label_particle_clusters(mapping)`` — e.g.
#       ``{0: "MODIFIER", 1: "CONNECTOR"}``, or whatever names the
#       actual cluster contents warrant (a cluster might not map to
#       either of those two pre-existing ideas at all — that's fine;
#       AGNN is allowed to surface a grammar class nobody anticipated).
#
# The feature vector (4 dimensions, all derived from data already
# computed during Pass 1 of train() — no new corpus pass needed):
#
#   1. pre_object_3tok_rate = pre_object_3tok_freq[tok] /
#      (pre_object_3tok_freq[tok] + pre_object_long_freq[tok])
#      -> high for tokens that sit at the pre-object slot mostly in
#         3-token sentences (the historical "MODIFIER" signature);
#         low for tokens that sit there mostly in long sentences (the
#         historical "CONNECTOR" signature). Tokens with zero
#         pre-object observations get 0.5 (neutral - no signal either
#         way), so they don't artificially cluster with either pole.
#
#   2. between_first_rate = corpus-wide between-first count /
#      total frequency
#      -> how often this token shows up specifically in the
#         between-action-and-object slot (vs elsewhere in the
#         sentence). Connectors score high here; modifiers, which sit
#         in the 3-token action slot rather than a >3-token
#         between-first slot, score lower.
#
#   3. fine_position_entropy = normalized Shannon entropy of
#      fine_positional_freq[tok] (already computed for function-word
#      discovery) -> how "spread out" the token's positions are.
#
#   4. bucket_entropy = normalized Shannon entropy of
#      positional_freq[tok] (coarse 4-bucket scheme) -> same idea at
#      the coarse-bucket level.
#
# Two particle tokens merge into the same cluster when the Euclidean
# distance between their (normalized 0..1) feature vectors is below
# ``_PARTICLE_DISTANCE_THRESHOLD``. This is a GENERIC distance
# metric over GENERIC statistics — it has no awareness of what
# "sangat" or "dari" mean, and no part of the threshold was reverse-
# engineered by checking where those specific tokens land. The
# threshold instead targets a natural separation in the data: the
# combined corpus's particle population is genuinely bimodal on
# dimension 1 (pre_object_3tok_rate clusters near 0.0 for one group
# and near 0.6-0.85 for another, with a near-empty gap around
# 0.2-0.5 -- see the calibration note on ``_PARTICLE_DISTANCE_THRESHOLD``
# for the actual gap measurement), so a threshold landing inside that
# gap separates the two natural groups regardless of which specific
# tokens happen to populate them.
_PARTICLE_DISTANCE_THRESHOLD = 0.35

# Minimum total frequency for a particle token to be clustered.
# Below this, the feature vector is too noisy (one observation can
# swing pre_object_3tok_rate from 0.0 to 1.0). Particle tokens below
# this floor stay in their own cluster_id (effectively unclustered;
# they get cluster_id = -1, same convention as unclustered actions).
_PARTICLE_MIN_FREQ = 3


# ----------------------------------------------------------------------
# Learner
# ----------------------------------------------------------------------

@dataclass
class PositionalClusterLearner:
    """Zero-bias positional cluster learner with post-hoc naming.

    Public API:
        train(corpus_lines)                  -> None
        inspect_cluster_details()            -> Dict[int, Dict[str, object]]
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
    # Statistical anchor-word discovery state (replaces _ACTION_STOPLIST
    # and _COPULAS). Populated by _compute_anchor_words() during train().
    # See the "_FUNCTION_WORD_*" and "_ACTION_ANCHOR_*" constants above
    # for the discovery contract.
    #
    # fine_positional_freq: {token: {fine_position: count}}
    #   Fine-grained positional counts where the position is the raw
    #   index 0..n-2 with -1 for the last token (NOT collapsed to the
    #   4-bucket scheme used by positional_freq). Used to compute the
    #   Shannon entropy of each token's position distribution; function
    #   words have flat (high-entropy) distributions, content words
    #   have concentrated (low-entropy) distributions.
    #
    # function_word_candidates: set of tokens statistically flagged as
    #   function words (high positional entropy + freq floor + no verb
    #   morphology). Excluded from the action slot during (action,
    #   object) extraction - the zero-bias replacement for the old
    #   hardcoded _ACTION_STOPLIST.
    #
    # action_bucket_anchors: set of tokens statistically flagged as
    #   action-bucket anchors (concentrated at the action bucket + freq
    #   floor + low bucket entropy). Recognised as valid actions in
    #   >3-token sentences even without verb morphology - the zero-bias
    #   replacement for the old hardcoded _COPULAS whitelist.
    # ------------------------------------------------------------------
    fine_positional_freq: Dict[str, Dict[int, int]] = field(default_factory=dict)
    function_word_candidates: Set[str] = field(default_factory=set)
    action_bucket_anchors: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Brown-clustering state (added by the object-vocabulary
    # super-cluster fix). See the "_BROWN_CLUSTER_*" constants above
    # for the algorithm contract.
    #
    # object_supercluster_id: {object_token: supercluster_id} — the
    #   flat assignment from each object token to its Brown super-cluster.
    #   Populated by _cluster_object_vocabulary() during train(), BEFORE
    #   action clustering. Action clustering then uses super-cluster
    #   distributions instead of literal object tokens.
    #
    # object_superclusters: {supercluster_id: set(object_tokens)} — the
    #   inverse map, for inspection/debugging. Exposed so callers can
    #   inspect which objects got grouped together.
    # ------------------------------------------------------------------
    object_supercluster_id: Dict[str, int] = field(default_factory=dict)
    object_superclusters: Dict[int, Set[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # MODIFIER (adverb / intensifier) discovery state.
    #
    # pre_object_3tok_freq: {token: count} — how often each token sits
    #   at index n-2 in 3-token sentences (= index 1 = action bucket).
    #   This is the "modifier of adjective" position. Built during
    #   Pass 1 of train().
    #
    # pre_object_long_freq: {token: count} — how often each token sits
    #   at index n-2 in >3-token sentences (= between-first slot).
    #   This is where CONNECTORS sit. Built during Pass 1 of train().
    #
    # The split by sentence length is the key signal: a MODIFIER's
    # dominant position is the 3-token action slot; a CONNECTOR's
    # dominant position is the >3-token between-first slot. See the
    # ``_MODIFIER_*`` constants above for the full contract.
    #
    # modifier_tokens: set of tokens statistically flagged as MODIFIERs
    #   (function_word_candidates whose pre_object_3tok_freq clears the
    #   count floor AND whose 3tok-rate clears the rate threshold).
    #   Exposed via tag_sentence() as a positive grammar class. Tokens
    #   classified as MODIFIER are REMOVED from connector_tokens
    #   (modifier classification takes priority — see _MODIFIER_* docs).
    # ------------------------------------------------------------------
    pre_object_3tok_freq: Dict[str, int] = field(default_factory=dict)
    pre_object_long_freq: Dict[str, int] = field(default_factory=dict)
    modifier_tokens: Set[str] = field(default_factory=set)

    # Corpus-wide between-first slot counts per token (any token, not
    # just connectors) — built by _compute_connector_signature() as a
    # byproduct of Phase A. Used by _compute_particle_signature() for
    # the between_first_rate feature dimension. Not persisted (cheap
    # to recompute on next train(); not needed for classify()/spo()).
    _between_first_counts: Dict[str, int] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # PARTICLE clustering state (zero-bias two-stage grammar-class
    # discovery — see the "PARTICLE clustering" section above for the
    # full rationale and the contrast with the MODIFIER/CONNECTOR
    # mechanism above, which bakes names into detection directly).
    #
    # particle_cluster_id_of: {token: cluster_id} — which UNNAMED
    #   particle cluster a token belongs to. -1 = below the frequency
    #   floor (unclustered, same convention as action cluster_id_of).
    #
    # particle_clusters: {cluster_id: set(tokens)} — the unnamed
    #   clusters themselves, formed purely by feature-vector distance.
    #
    # particle_cluster_labels: {cluster_id: str} — human-assigned
    #   names, populated ONLY by label_particle_clusters(). Empty
    #   until a human reviews inspect_particle_clusters() and decides.
    # ------------------------------------------------------------------
    particle_cluster_id_of: Dict[str, int] = field(default_factory=dict)
    particle_clusters: Dict[int, Set[str]] = field(default_factory=dict)
    particle_cluster_labels: Dict[int, str] = field(default_factory=dict)

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

        Pipeline (two-pass, zero-bias):
            Pass 1 - position statistics only:
              1. Parse each line into tokens. For every token at every
                 position, bump BOTH ``positional_freq`` (4-bucket scheme:
                 0/1/2/-1) AND ``fine_positional_freq`` (raw index, -1
                 for last). The fine-grained scheme powers the Shannon
                 entropy calculation that distinguishes function words
                 (flat distribution) from content words (concentrated).

            Anchor-word discovery (between passes):
              2. Compute ``function_word_candidates`` from
                 ``fine_positional_freq``: tokens whose position
                 distribution is flat (high normalized entropy),
                 frequent (>= ``_FUNCTION_WORD_MIN_FREQ``), span >=
                 ``_FUNCTION_WORD_MIN_POSITIONS`` distinct positions,
                 and lack verb morphology. These are statistically
                 discovered function words - the zero-bias replacement
                 for the old hardcoded ``_ACTION_STOPLIST``.
              3. Compute ``action_bucket_anchors`` from
                 ``positional_freq``: tokens concentrated at the action
                 bucket (low bucket entropy), frequent (>=
                 ``_ACTION_ANCHOR_MIN_FREQ``), with the action bucket
                 as their dominant bucket. These are statistically
                 discovered action anchors - the zero-bias replacement
                 for the old hardcoded ``_COPULAS`` whitelist. Copulas
                 like 'adalah' and 'merupakan' emerge here automatically.

            Pass 2 - (action, object) extraction using discovered sets:
              4. Re-iterate the corpus. For each SVO-shaped sentence
                 (>= 3 tokens), extract (action, object) using
                 ``self.function_word_candidates`` (excluded from action
                 slot) and ``self.action_bucket_anchors`` (recognised as
                 verbs even without morphology). Bump
                 ``action_object_freq`` and record the "between-first"
                 token for the connector-signal detector.

            Post-processing (unchanged from v2):
              5. Compute the connector signature
                 (``action_connector_signature`` + corpus-wide
                 ``connector_tokens``) from the between-first
                 observations.
              6. Brown-cluster the object vocabulary (NEW):
                 ``_cluster_object_vocabulary`` groups object tokens
                 by their action-context distributions, producing
                 ``object_supercluster_id`` and ``object_superclusters``.
              7. Cluster actions by weighted Jaccard of their
                 super-cluster distributions (NEW: was literal object
                 tokens), split first by has_connector.
              8. Reset cluster_labels (labelling must be redone after
                 re-training because cluster_ids may shift).

        Failure contract: empty / single-token lines are skipped
        silently. A train() call with zero usable lines leaves the
        learner un-trained; classify() then delegates to fallback.
        """
        if not corpus_lines:
            return

        # ----------------------------------------------------------------
        # PASS 1 - position statistics only (no action/object extraction
        # yet; we need the populated frequency tables to discover
        # function words and action anchors first).
        # ----------------------------------------------------------------
        # Cache the tokenised corpus so Pass 2 doesn't re-tokenise.
        # Memory cost is bounded by corpus size; pretrain_corpus is
        # ~3500 lines so this is fine.
        tokenised_lines: List[List[str]] = []
        for line in corpus_lines:
            tokens = self._tokenize(line)
            if not tokens:
                tokenised_lines.append([])  # preserve index alignment
                continue
            tokenised_lines.append(tokens)
            n = len(tokens)
            buckets = self._compute_buckets(n)
            for i, token in enumerate(tokens):
                # Coarse bucket (existing 4-bucket scheme).
                b = buckets[i]
                pos_map = self.positional_freq.setdefault(token, {})
                pos_map[b] = pos_map.get(b, 0) + 1
                # Fine-grained position: raw index, -1 for last.
                # Used for entropy-based function word discovery.
                fi = i if i != n - 1 else -1
                fine_map = self.fine_positional_freq.setdefault(token, {})
                fine_map[fi] = fine_map.get(fi, 0) + 1
            # Pre-object slot count (for MODIFIER discovery — see
            # _MODIFIER_* constants above). The token at index n-2 is
            # the position immediately before the last-token object.
            # We split by sentence length:
            #   - 3-token sentences: index n-2 = index 1 = action
            #     bucket. This is where MODIFIERs sit in "state + adj"
            #     sentences ("es SANGAT dingin").
            #   - >3-token sentences: index n-2 = between-first slot.
            #     This is where CONNECTORS sit ("X berbeda DARI Y").
            # The split is the key signal for distinguishing the two
            # grammar classes — see _compute_modifiers().
            if n >= 3:
                pre_obj_tok = tokens[n - 2]
                if n == 3:
                    self.pre_object_3tok_freq[pre_obj_tok] = (
                        self.pre_object_3tok_freq.get(pre_obj_tok, 0) + 1
                    )
                else:
                    self.pre_object_long_freq[pre_obj_tok] = (
                        self.pre_object_long_freq.get(pre_obj_tok, 0) + 1
                    )

        # ----------------------------------------------------------------
        # Anchor-word discovery (between passes).
        # ----------------------------------------------------------------
        self._compute_anchor_words()

        # ----------------------------------------------------------------
        # Preliminary MODIFIER discovery (between passes, BEFORE Pass 2).
        # ----------------------------------------------------------------
        # Detect modifier tokens from pre_object_3tok_freq so they can
        # be excluded from action extraction in Pass 2 (same as
        # function_word_candidates). The connector priority rule
        # (removing MODIFIERs from connector_tokens) is applied later,
        # after _compute_connector_signature().
        self._compute_modifiers()

        # ----------------------------------------------------------------
        # PASS 2 - (action, object) extraction using discovered sets.
        # ----------------------------------------------------------------
        # Per-action between-first observations. Each entry is
        # {action: {between_first_token_or_None: count}}. We accumulate
        # this alongside action_object_freq so the connector detector
        # has the raw positional evidence it needs without re-parsing
        # the corpus.
        action_between: Dict[str, Dict[Optional[str], int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for tokens in tokenised_lines:
            if not tokens or len(tokens) < 3:
                continue
            action_token, object_token = self._extract_action_object(tokens)
            if action_token and object_token:
                obj_bucket = self.action_object_freq.setdefault(action_token, {})
                obj_bucket[object_token] = obj_bucket.get(object_token, 0) + 1
                # Connector-signal evidence: record the token (or
                # None) that sits immediately after the action and
                # before the object.
                between_token = self._extract_between_token(
                    tokens, action_token, object_token
                )
                action_between[action_token][between_token] += 1

        # ----------------------------------------------------------------
        # Post-processing.
        # ----------------------------------------------------------------
        # Connector signature (existing).
        self._compute_connector_signature(action_between)

        # MODIFIER connector-priority rule: now that connector_tokens
        # is populated, remove any MODIFIERs from it. The preliminary
        # MODIFIER detection ran before Pass 2 (to exclude modifiers
        # from action extraction); here we just apply the priority
        # rule. We do NOT re-run _compute_modifiers() because that
        # would use the now-populated action_object_freq (which
        # includes modifiers that slipped through Pass 2 before the
        # preliminary detection — but the preliminary detection
        # already caught them, so action_object_freq should not
        # contain modifier tokens).
        self.connector_tokens -= self.modifier_tokens

        # PARTICLE clustering (zero-bias two-stage grammar-class
        # discovery): cluster the leftover particle population by
        # feature-vector distance, unnamed. Must run after
        # function_word_candidates / connector_tokens / modifier_tokens
        # / _between_first_counts are all populated.
        self._cluster_particles()

        # Brown-cluster the object vocabulary BEFORE action clustering,
        # so action clustering can use super-cluster distributions.
        self._cluster_object_vocabulary()

        # Cluster actions by similarity of super-cluster distributions,
        # split first by has_connector.
        self._cluster_actions()

        # Reset labels (cluster_ids may have shifted; old labels are
        # no longer meaningful). The human must call label_clusters()
        # again after re-training.
        self.cluster_labels = {}

    def _cluster_actions(self) -> None:
        """Greedy agglomerative clustering of actions by weighted Jaccard.

        Algorithm (pure Python, no sklearn / scipy):
            1. Build the set of "clusterable" actions: those with at
               least ``min_action_observations`` total co-occurrence
               counts.
            2. **Super-cluster projection** (NEW): for each clusterable
               action, project its literal object-token count map to a
               SUPER-CLUSTER count map via
               ``self.object_supercluster_id``. Two actions whose
               literal object tokens never overlap can still merge if
               their objects belong to the same Brown super-cluster
               (e.g. 'adalah' taking mamalia/logam and 'merupakan'
               taking mamalia/unggas merge because mamalia, logam, and
               unggas all belong to the 'taxonomy noun' super-cluster).
            3. **Connector split**: partition the clusterable actions
               into two groups by their
               ``action_connector_signature`` value — has_connector=True
               vs has_connector=False. The two groups are clustered
               INDEPENDENTLY and can never merge across the split.
               This is the structural-signal fix for cluster 62:
               "adalah" (direct object) and "berbeda" (connector +
               object) end up in different clusters even though their
               object distributions overlap on taxonomy nouns.
            4. Initialise each cluster as a singleton {action} with
               BOTH its literal object count map AND its super-cluster
               count map.
            5. Greedy pass: for every pair of clusters *within the
               same connector group*, compute similarity as the MAX
               of (a) literal weighted Jaccard and (b) super-cluster
               weighted Jaccard. This preserves the existing
               literal-token-based clustering behaviour (so the 5
               RelationType clusters still form) while ALSO allowing
               synonym copulas to merge via super-cluster overlap
               when their literal overlap is insufficient (the
               Brown-clustering fix). If similarity >= threshold,
               merge them.
            6. Repeat passes until no merge happens (fixpoint).
            7. Assign cluster_ids (0, 1, 2, ...) across both groups
               (no-connector group first, then with-connector group).
               Actions that did not meet the min_observations bar get
               cluster_id = -1 (unclustered).

        Why MAX of literal and super-cluster similarities (not
        super-cluster alone)? Super-cluster projection loses
        information: two actions with similar SUPER-cluster
        distributions might still take SEMANTICALLY DIFFERENT objects
        (e.g. 'menyebabkan' takes state-change objects and 'setelah'
        takes event objects, but both object classes can land in the
        same Brown super-cluster because they co-occur with overlapping
        action sets). Using super-cluster alone causes cross-relation-
        type merging (CAUSAL+TEMPORAL collapse into one cluster).

        Taking the MAX preserves the existing literal-token behaviour:
        actions that already merge via literal overlap (e.g.
        'adalah'+'merupakan' sharing 'mamalia') still merge. The
        super-cluster signal only ADDS new merges for actions whose
        literal overlap is below threshold but whose super-cluster
        overlap is high (e.g. synonyms with no literal object overlap
        but whose objects all belong to the same Brown super-cluster).
        """
        # Reset previous clustering.
        self.cluster_id_of = {}
        self.action_clusters = {}

        # Build the set of clusterable actions + their object count maps.
        clusterable: Dict[str, Dict[str, int]] = {}
        for action, objs in self.action_object_freq.items():
            total = sum(objs.values())
            if total >= self.min_action_observations:
                clusterable[action] = dict(objs)

        # Super-cluster projection: convert each action's literal
        # {object_token: count} map to a {supercluster_id: count} map.
        clusterable_sc: Dict[str, Dict[int, int]] = {}
        for action, objs in clusterable.items():
            sc_map: Dict[int, int] = defaultdict(int)
            for obj, count in objs.items():
                sc_id = self.object_supercluster_id.get(obj)
                if sc_id is None:
                    # No super-cluster info — use a stable hash of the
                    # object token as a fallback super-cluster id so
                    # distinct objects stay distinct.
                    sc_id = hash(obj)
                sc_map[sc_id] += count
            clusterable_sc[action] = dict(sc_map)

        # Connector split: partition clusterable actions by their
        # action_connector_signature value.
        no_connector_lit: Dict[str, Dict[str, int]] = {}
        no_connector_sc: Dict[str, Dict[int, int]] = {}
        with_connector_lit: Dict[str, Dict[str, int]] = {}
        with_connector_sc: Dict[str, Dict[int, int]] = {}
        for action in clusterable:
            if self.action_connector_signature.get(action, False):
                with_connector_lit[action] = clusterable[action]
                with_connector_sc[action] = clusterable_sc[action]
            else:
                no_connector_lit[action] = clusterable[action]
                no_connector_sc[action] = clusterable_sc[action]

        # Cluster each group independently.
        all_clusters: List[Tuple[Set[str], Dict[str, int], Dict[int, int]]] = []
        for lit_group, sc_group in (
            (no_connector_lit, no_connector_sc),
            (with_connector_lit, with_connector_sc),
        ):
            group_clusters = self._cluster_action_group(lit_group, sc_group)
            all_clusters.extend(group_clusters)

        # Assign cluster_ids across the concatenated group clusters.
        for cluster_id, (actions, _lit, _sc) in enumerate(all_clusters):
            self.action_clusters[cluster_id] = actions
            for action in actions:
                self.cluster_id_of[action] = cluster_id

        # Mark unclustered actions with cluster_id = -1.
        for action in self.action_object_freq:
            if action not in self.cluster_id_of:
                self.cluster_id_of[action] = -1

    def _cluster_action_group(
        self,
        actions_lit: Dict[str, Dict[str, int]],
        actions_sc: Dict[str, Dict[int, int]],
    ) -> List[Tuple[Set[str], Dict[str, int], Dict[int, int]]]:
        """Run the greedy agglomerative merge on one connector group.

        Args:
            actions_lit: {action_token: {object_token: count}} —
                the literal object distribution for each action.
            actions_sc: {action_token: {supercluster_id: count}} —
                the projected super-cluster distribution for each
                action.

        Returns a list of (set_of_actions, literal_count_map,
        supercluster_count_map) — the merged clusters for this group.
        Empty list if the group has no actions.

        Similarity = MAX of literal weighted Jaccard and super-cluster
        weighted Jaccard. See ``_cluster_actions`` for the rationale.
        """
        if not actions_lit:
            return []

        # Initial clusters: each action in its own cluster.
        clusters: List[Tuple[Set[str], Dict[str, int], Dict[int, int]]] = [
            ({action}, dict(actions_lit[action]), dict(actions_sc[action]))
            for action in actions_lit
        ]

        # Greedy agglomerative merge until fixpoint.
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            best_i, best_j, best_sim = -1, -1, -1.0
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    lit_sim = self._weighted_jaccard(
                        clusters[i][1], clusters[j][1]
                    )
                    sc_sim = self._weighted_jaccard(
                        clusters[i][2], clusters[j][2]
                    )
                    sim = max(lit_sim, sc_sim)
                    if sim >= self.similarity_threshold and sim > best_sim:
                        best_sim = sim
                        best_i, best_j = i, j
            if best_i >= 0:
                actions_i, lit_i, sc_i = clusters[best_i]
                actions_j, lit_j, sc_j = clusters[best_j]
                actions_i.update(actions_j)
                for obj, count in lit_j.items():
                    lit_i[obj] = lit_i.get(obj, 0) + count
                for sc_id, count in sc_j.items():
                    sc_i[sc_id] = sc_i.get(sc_id, 0) + count
                clusters.pop(best_j)
                merged = True

        return clusters

    # ------------------------------------------------------------------
    # Statistical anchor-word discovery (replaces _ACTION_STOPLIST/_COPULAS)
    # ------------------------------------------------------------------

    def _compute_anchor_words(self) -> None:
        """Populate ``function_word_candidates`` and ``action_bucket_anchors``.

        Called between Pass 1 and Pass 2 of train(), after
        ``positional_freq`` and ``fine_positional_freq`` are populated.
        See the ``_FUNCTION_WORD_*`` and ``_ACTION_ANCHOR_*`` constants
        for the discovery contract.

        Reset contract: this method overwrites both fields from
        scratch, so it is safe to call on every train() (no stale
        entries from a previous corpus survive).
        """
        self.function_word_candidates = set()
        self.action_bucket_anchors = set()

        # Function word candidates: scan fine_positional_freq AND
        # positional_freq (need both — fine for entropy, bucket for
        # concentration check).
        for token, fine_map in self.fine_positional_freq.items():
            total = sum(fine_map.values())
            if total < _FUNCTION_WORD_MIN_FREQ:
                continue
            if len(fine_map) < _FUNCTION_WORD_MIN_POSITIONS:
                continue
            # Verb-morphology tokens are content words by form, not
            # function words — skip even if their entropy is high.
            if self._looks_like_verb(token):
                continue
            # Normalized Shannon entropy over fine positions.
            h = 0.0
            for c in fine_map.values():
                if c > 0:
                    p = c / total
                    h -= p * math.log2(p)
            max_h = math.log2(len(fine_map)) if len(fine_map) > 1 else 0.0
            nh = h / max_h if max_h > 0 else 0.0
            if nh < _FUNCTION_WORD_ENTROPY_THRESHOLD:
                continue
            # Bucket concentration check: real function words (sangat,
            # itu, bukan, tidak, ...) concentrate at the action bucket
            # (b_nh near 0). Content words that span roles (api as
            # subject+object, makan as verb+subject+object) have high
            # bucket entropy. Requiring low bucket entropy excludes
            # these false positives.
            bucket_map = self.positional_freq.get(token, {})
            b_total = sum(bucket_map.values())
            if b_total == 0:
                continue
            b_h = 0.0
            for c in bucket_map.values():
                if c > 0:
                    p = c / b_total
                    b_h -= p * math.log2(p)
            b_max_h = (
                math.log2(len(bucket_map)) if len(bucket_map) > 1 else 0.0
            )
            b_nh = b_h / b_max_h if b_max_h > 0 else 0.0
            if b_nh >= _FUNCTION_WORD_MAX_BUCKET_ENTROPY:
                continue
            self.function_word_candidates.add(token)

        # Action bucket anchors: scan positional_freq.
        for token, bucket_map in self.positional_freq.items():
            total = sum(bucket_map.values())
            if total < _ACTION_ANCHOR_MIN_FREQ:
                continue
            # Find the dominant bucket (highest count, ties broken by
            # lowest bucket number for determinism).
            dominant_bucket = max(
                bucket_map.keys(),
                key=lambda b: (bucket_map[b], -b)
            )
            if dominant_bucket != _ACTION_BUCKET:
                continue
            # Normalized Shannon entropy over buckets.
            h = 0.0
            for c in bucket_map.values():
                if c > 0:
                    p = c / total
                    h -= p * math.log2(p)
            max_h = math.log2(len(bucket_map)) if len(bucket_map) > 1 else 0.0
            nh = h / max_h if max_h > 0 else 0.0
            if nh < _ACTION_ANCHOR_MAX_BUCKET_ENTROPY:
                self.action_bucket_anchors.add(token)

    # ------------------------------------------------------------------
    # Brown clustering of object vocabulary (Task 2)
    # ------------------------------------------------------------------

    def _cluster_object_vocabulary(self) -> None:
        """Brown-cluster the object vocabulary by action-context distribution.

        Populates ``object_supercluster_id`` and ``object_superclusters``
        from ``action_object_freq``. Called after Pass 2 of train() and
        BEFORE ``_cluster_actions()`` so action clustering can use
        super-cluster distributions instead of literal object tokens.

        Algorithm (pure Python, no sklearn / scipy):
            1. Build the object vocabulary: every token that appears
               as an object of any action in ``action_object_freq``.
            2. For each object token, build its "action context
               distribution" = {action_token: count} aggregated across
               all (action, object) pairs.
            3. **Inverted index**: build action -> {objects that
               co-occur with it}. Two objects are CANDIDATES for
               merging only if they share at least one action. This
               prunes the O(N^2) all-pairs scan down to the
               actually-comparable pairs (typically O(N) for sparse
               co-occurrence graphs).
            4. Greedy agglomerative merge: start with each object in
               its own cluster. Repeatedly find the candidate pair
               with the highest *weighted* Jaccard similarity of
               their action-context COUNT maps. If similarity >=
               ``_BROWN_CLUSTER_SIMILARITY_THRESHOLD``, merge them.
               Stop when no pair qualifies or the cluster count drops
               to ``_BROWN_CLUSTER_MAX_CLUSTERS``.
            5. Assign super-cluster ids (0, 1, 2, ...) and build both
               ``object_supercluster_id`` (forward map) and
               ``object_superclusters`` (inverse map).

        Why *weighted* Jaccard on count maps (not plain Jaccard on
        sets)? Plain Jaccard suffers from chain-merging: two objects
        that share ONE action merge at Jaccard >= 0.13, then the
        merged cluster's action set grows, enabling further merges
        via different shared actions. The end result is one giant
        super-cluster containing most of the object vocabulary.

        Weighted Jaccard is more strict: it considers COUNT
        distributions, not just set membership. Two objects that
        share an action but at very different frequencies get a lower
        similarity than two objects with matching count shapes. This
        breaks the chain-merge: an object can merge with the growing
        cluster only if its count distribution matches the cluster's
        aggregated distribution, not just shares one action.

        Reset contract: this method overwrites both fields from
        scratch, so it is safe to call on every train() (no stale
        entries from a previous corpus survive).
        """
        self.object_supercluster_id = {}
        self.object_superclusters = {}

        if not self.action_object_freq:
            return

        # Step 1+2: build object vocabulary and action-context distributions.
        # object_actions[obj] = {action: count}
        object_actions: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for action, objs in self.action_object_freq.items():
            for obj, count in objs.items():
                object_actions[obj][action] += count

        if not object_actions:
            return

        # Step 3: inverted index — action -> set of object indices that
        # co-occur with it. Used to enumerate candidate pairs (objects
        # that share at least one action) without scanning all O(N^2)
        # pairs.
        obj_list: List[str] = list(object_actions.keys())
        obj_to_idx: Dict[str, int] = {obj: i for i, obj in enumerate(obj_list)}
        action_to_obj_indices: Dict[str, Set[int]] = defaultdict(set)
        for obj, actions in object_actions.items():
            for action in actions:
                action_to_obj_indices[action].add(obj_to_idx[obj])

        # Step 4: greedy agglomerative merge using candidate pairs.
        # Each cluster is (set_of_object_indices, dict_of_action_counts).
        # Track cluster-level action -> set of cluster indices for
        # candidate generation.
        clusters: List[Tuple[Set[int], Dict[str, int]]] = [
            ({i}, dict(object_actions[obj]))
            for i, obj in enumerate(obj_list)
        ]
        # cluster_idx_for_obj: object idx -> cluster idx that contains it.
        cluster_idx_for_obj: List[int] = list(range(len(obj_list)))
        # action -> set of cluster indices that contain at least one
        # object co-occurring with this action. Updated as clusters merge.
        action_to_cluster_indices: Dict[str, Set[int]] = {
            action: set(action_to_obj_indices[action])
            for action in action_to_obj_indices
        }

        threshold = _BROWN_CLUSTER_SIMILARITY_THRESHOLD
        max_clusters = _BROWN_CLUSTER_MAX_CLUSTERS

        merged = True
        while merged and len(clusters) > max(1, max_clusters):
            merged = False
            # Enumerate candidate pairs: clusters that share at least
            # one action.
            candidate_set: Set[Tuple[int, int]] = set()
            for cluster_indices in action_to_cluster_indices.values():
                cluster_list = sorted(cluster_indices)
                for i in range(len(cluster_list)):
                    for j in range(i + 1, len(cluster_list)):
                        ci, cj = cluster_list[i], cluster_list[j]
                        if ci < cj:
                            candidate_set.add((ci, cj))
                        else:
                            candidate_set.add((cj, ci))

            if not candidate_set:
                break

            best_i, best_j, best_sim = -1, -1, -1.0
            for ci, cj in candidate_set:
                sim = self._weighted_jaccard(
                    clusters[ci][1], clusters[cj][1]
                )
                if sim >= threshold and sim > best_sim:
                    best_sim = sim
                    best_i, best_j = ci, cj

            if best_i >= 0:
                # Merge cluster best_j into cluster best_i.
                objs_i, acts_i = clusters[best_i]
                objs_j, acts_j = clusters[best_j]
                objs_i.update(objs_j)
                for act, count in acts_j.items():
                    acts_i[act] = acts_i.get(act, 0) + count
                # Update cluster_idx_for_obj for objects in best_j.
                for obj_idx in objs_j:
                    cluster_idx_for_obj[obj_idx] = best_i
                # Remove best_j from action_to_cluster_indices.
                for act in acts_j:
                    action_to_cluster_indices[act].discard(best_j)
                    if best_i not in action_to_cluster_indices[act]:
                        action_to_cluster_indices[act].add(best_i)
                # Mark cluster best_j as empty (will be filtered out
                # at the end). We use a sentinel rather than removing
                # from clusters list to keep indices stable.
                clusters[best_j] = (set(), {})
                merged = True

        # Step 5: assign super-cluster ids and build inverse map.
        # Filter out empty clusters (merged into others).
        sc_id = 0
        for cluster_objs, _acts in clusters:
            if not cluster_objs:
                continue
            obj_set = {obj_list[i] for i in cluster_objs}
            self.object_superclusters[sc_id] = obj_set
            for obj in obj_set:
                self.object_supercluster_id[obj] = sc_id
            sc_id += 1

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

        # Stash the raw between-first counts (any token, not just
        # qualifying connectors) for _compute_particle_signature()'s
        # between_first_rate dimension.
        self._between_first_counts = dict(between_first_counts)

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

    # ------------------------------------------------------------------
    # MODIFIER (adverb / intensifier) discovery
    # ------------------------------------------------------------------

    def _compute_modifiers(self) -> None:
        """Populate ``modifier_tokens`` from positional evidence.

        Called during train() after :meth:`_compute_anchor_words`
        (which populates ``function_word_candidates``) but BEFORE
        Pass 2 (action/object extraction). This ordering is critical:
        modifier tokens must be identified before action extraction
        so they can be excluded from the action slot (same as
        ``function_word_candidates``). Without this, modifiers like
        'sangat' would be extracted as actions in 3-token "state +
        adj" sentences ("es sangat dingin" → action='sangat'), which
        would pollute ``action_object_freq`` and ``cluster_id_of``.

        See the ``_MODIFIER_*`` constants above for the full contract.

        Reset contract: this method overwrites ``modifier_tokens`` from
        scratch, so it is safe to call on every train() (no stale
        entries from a previous corpus survive).

        Connector priority rule: this method does NOT remove
        MODIFIERs from ``connector_tokens`` because
        ``connector_tokens`` is not yet populated at this point in
        train(). The priority rule is applied separately after
        :meth:`_compute_connector_signature` runs.

        Zero-bias contract: no hardcoded adverb list. The detection
        uses only positional + frequency statistics + verb-morphology
        heuristic, same as the anchor-word discovery.
        """
        self.modifier_tokens = set()

        # Pre-object freq tables were built during Pass 1 of train().
        # If empty (e.g. tiny corpus with no >=3-token sentences), no
        # modifiers can be discovered — leave the set empty.
        if not self.pre_object_3tok_freq:
            return

        # Scan tokens in function_word_candidates (statistically
        # flagged as non-action by the anchor-word discovery — high
        # positional entropy + freq floor + no verb morphology).
        # This is the zero-bias replacement for a hardcoded adverb
        # list: function_word_candidates already excludes copulas
        # like 'adalah' (low entropy — always at action slot) and
        # real verbs (verb morphology check). Modifiers like 'sangat'
        # (high entropy — appears at multiple positions) are IN
        # function_word_candidates.
        #
        # We do NOT scan all tokens with pre_object_3tok_freq because
        # that would include copulas ('adalah', 'merupakan') and
        # perceptual predicates ('tampak', 'terasa') that sit at the
        # action slot in 3-token sentences but are real actions, not
        # modifiers. The function_word_candidates filter correctly
        # excludes these (they have low positional entropy).
        for token in self.function_word_candidates:
            pre_3tok = self.pre_object_3tok_freq.get(token, 0)
            # Count floor: need enough 3-token pre-object observations
            # to be statistically confident this is a modifier pattern.
            if pre_3tok < _MODIFIER_MIN_PRE_OBJECT_COUNT:
                continue
            pre_long = self.pre_object_long_freq.get(token, 0)
            total_pre = pre_3tok + pre_long
            if total_pre == 0:
                continue
            # 3tok rate: the fraction of pre-object observations that
            # come from 3-token sentences. A MODIFIER's dominant
            # position is the 3-token action slot; a CONNECTOR's
            # dominant position is the >3-token between-first slot.
            # This ratio cleanly separates the two.
            rate_3tok = pre_3tok / total_pre
            if rate_3tok < _MODIFIER_3TOK_RATE:
                continue
            self.modifier_tokens.add(token)

    # ------------------------------------------------------------------
    # PARTICLE clustering: zero-bias two-stage grammar-class discovery
    # ------------------------------------------------------------------

    def _compute_particle_signature(self, token: str) -> Tuple[float, float, float, float]:
        """Build a 4-dim positional feature vector for ``token``.

        Returns ``(pre_object_3tok_rate, between_first_rate,
        fine_position_entropy, bucket_entropy)``, each normalized to
        ``[0.0, 1.0]``. See the "PARTICLE clustering" module section
        for the full definition of each dimension. No part of this
        computation references what the token means or is "supposed"
        to be classified as — it is purely a function of corpus
        position counts already gathered during Pass 1 of train().
        """
        pre_3tok = self.pre_object_3tok_freq.get(token, 0)
        pre_long = self.pre_object_long_freq.get(token, 0)
        total_pre = pre_3tok + pre_long
        # Neutral midpoint (0.5) when there's no pre-object signal at
        # all, so tokens with zero observations on this dimension
        # don't get pulled toward either pole by default.
        pre_object_3tok_rate = pre_3tok / total_pre if total_pre > 0 else 0.5

        fine_map = self.fine_positional_freq.get(token, {})
        fine_total = sum(fine_map.values())
        between_first_rate = 0.0
        if fine_total > 0:
            # Approximate "between-first" exposure using the
            # corpus-wide between_first count already computed for
            # the connector detector, normalized by this token's
            # total frequency.
            between_count = self._between_first_counts.get(token, 0)
            between_first_rate = min(1.0, between_count / fine_total)

        def _normalized_entropy(freq_map: Dict[int, int]) -> float:
            total = sum(freq_map.values())
            if total <= 0 or len(freq_map) <= 1:
                return 0.0
            h = 0.0
            for c in freq_map.values():
                if c > 0:
                    p = c / total
                    h -= p * math.log2(p)
            max_h = math.log2(len(freq_map))
            return h / max_h if max_h > 0 else 0.0

        fine_entropy = _normalized_entropy(fine_map)
        bucket_entropy = _normalized_entropy(self.positional_freq.get(token, {}))

        return (pre_object_3tok_rate, between_first_rate, fine_entropy, bucket_entropy)

    def _cluster_particles(self) -> None:
        """Cluster particle tokens by feature-vector distance (unnamed).

        STAGE 1 of the zero-bias particle discovery contract (see the
        module-level "PARTICLE clustering" section). Operates on the
        union of ``function_word_candidates`` and ``connector_tokens``
        candidates — i.e. every token already statistically excluded
        from being a real content action — plus any token that was
        classified MODIFIER by the legacy mechanism. This is the
        "leftover" particle population: tokens whose role is
        grammatical, not lexical-content.

        No RelationType, "MODIFIER", or "CONNECTOR" string is ever
        consulted during this method. Clusters are pure integer ids.
        A human must call :meth:`label_particle_clusters` afterward to
        assign names — that is a SEPARATE, optional, post-hoc step
        (mirroring :meth:`label_clusters` for RelationType).

        Reset contract: overwrites particle_cluster_id_of and
        particle_clusters from scratch on every call (cluster ids may
        shift between train() calls, same as action cluster_id_of).
        particle_cluster_labels is also reset, because labels assigned
        to old cluster_ids are not meaningful for new ones — same
        contract as cluster_labels being reset in train().
        """
        self.particle_cluster_id_of = {}
        self.particle_clusters = {}
        self.particle_cluster_labels = {}

        candidates: Set[str] = (
            set(self.function_word_candidates)
            | set(self.connector_tokens)
            | set(self.modifier_tokens)
        )
        # Exclude tokens that are already established as real content
        # actions (have their own action_object_freq entry — i.e. they
        # ARE the predicate of at least one sentence, with their own
        # object distribution). This is a role-priority exclusion, not
        # a semantic one: a token already doing duty as a content verb
        # shouldn't also compete for a "leftover particle" cluster.
        # Without this, verbs that incidentally sit in a between-first
        # slot a handful of times (a latent looseness in the connector
        # detector's "never appears as object" check — true of nearly
        # every verb, since verbs are not nouns) leak into the particle
        # pool and pollute clusters with real content words.
        candidates -= set(self.action_object_freq.keys())
        if not candidates:
            return

        # Frequency floor: tokens with too few observations get an
        # unreliable feature vector. They stay unclustered (-1).
        def _total_freq(tok: str) -> int:
            return sum(self.fine_positional_freq.get(tok, {}).values())

        clusterable = sorted(t for t in candidates if _total_freq(t) >= _PARTICLE_MIN_FREQ)
        for tok in candidates:
            if tok not in clusterable:
                self.particle_cluster_id_of[tok] = -1

        if not clusterable:
            return

        signatures: Dict[str, Tuple[float, float, float, float]] = {
            tok: self._compute_particle_signature(tok) for tok in clusterable
        }

        # Greedy agglomerative merge by Euclidean distance over the
        # 4-dim feature vector (mean-aggregated per cluster). Pure
        # generic distance — no semantic awareness of any token.
        clusters: List[Tuple[Set[str], Tuple[float, float, float, float]]] = [
            ({tok}, sig) for tok, sig in signatures.items()
        ]

        def _distance(
            a: Tuple[float, float, float, float],
            b: Tuple[float, float, float, float],
        ) -> float:
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

        merged = True
        while merged and len(clusters) > 1:
            merged = False
            best_i, best_j, best_dist = -1, -1, float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = _distance(clusters[i][1], clusters[j][1])
                    if d <= _PARTICLE_DISTANCE_THRESHOLD and d < best_dist:
                        best_dist = d
                        best_i, best_j = i, j
            if best_i >= 0:
                toks_i, sig_i = clusters[best_i]
                toks_j, sig_j = clusters[best_j]
                merged_toks = toks_i | toks_j
                # Recompute the cluster's representative signature as
                # the mean of its members' individual signatures
                # (centroid), not the mean of the two merging
                # sub-clusters' signatures, to avoid drift bias toward
                # whichever sub-cluster was larger.
                member_sigs = [signatures[t] for t in merged_toks]
                centroid = tuple(
                    sum(s[d] for s in member_sigs) / len(member_sigs)
                    for d in range(4)
                )
                clusters[best_i] = (merged_toks, centroid)
                clusters.pop(best_j)
                merged = True

        for cluster_id, (toks, _centroid) in enumerate(clusters):
            self.particle_clusters[cluster_id] = toks
            for tok in toks:
                self.particle_cluster_id_of[tok] = cluster_id

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

    # ------------------------------------------------------------------
    # Public API: synthetic sentence generation (for RLHF-style feedback loop)
    # ------------------------------------------------------------------

    def sample_sentence(
        self,
        cluster_id: int,
        *,
        rng: Optional[Any] = None,
    ) -> Optional[str]:
        """Generate one synthetic SVO sentence from a learned cluster.

        Picks an action from the cluster, picks the object most
        statistically associated with that action (from
        ``action_object_freq``), and picks a subject from tokens that
        frequently appear at the agent position (bucket 0) anywhere
        in the corpus. Returns the sentence as ``"Subject Action
        Object"`` (lower-cased, single-space separated).

        Why this is the synthetic-generation contract:

          * **Action** — sampled uniformly from the cluster's action
            set so every cluster member gets airtime. The action
            token is what ``classify()`` will look up to label the
            sentence, so it MUST be a real cluster member.

          * **Object** — picked as the HIGHEST-count object in
            ``action_object_freq[action]``. Statistical association:
            the object the corpus most frequently paired with this
            action. We don't sample-weight because we want the
            generated sentence to be maximally representative of the
            action's observed distribution — the user's verdict
            ("makes sense" / "doesn't make sense") should reflect the
            cluster's CENTRAL tendency, not a tail co-occurrence.

          * **Subject** — picked from tokens whose
            ``positional_freq[token][0]`` (agent bucket) count is
            highest. This is corpus-wide, not action-specific — we
            don't track which subjects co-occurred with which actions
            (the cluster learner only stores action↔object
            co-occurrence). The subject is a "carrier" token that
            makes the sentence grammatical; the SEMANTIC content the
            user evaluates is the action↔object pair. Subject
            selection from the agent-bucket distribution keeps the
            generated sentence corpus-grounded (real subject tokens
            the learner has seen) without biasing the verdict toward
            any particular subject.

        Zero-bias contract: this method does NOT consult any human
        word list. Action, object, and subject are all picked from
        learned statistical distributions. The generated sentence is
        a faithful projection of what the cluster's central tendency
        looks like in surface form — if the user's verdict is "bad",
        the cluster's central tendency is misaligned with the user's
        semantic expectation, and ``AGNNCore.apply_feedback`` will
        adjust the action↔object edge weight accordingly (NOT relabel
        the cluster).

        Args:
            cluster_id: The cluster to sample from. Must be a real
                cluster id in ``self.action_clusters`` (i.e. >= 0 and
                present as a key). -1 (unclustered) is rejected.
            rng: Optional random.Random instance for reproducible
                sampling. When ``None`` (the default), a fresh
                ``random.Random()`` is constructed per call — callers
                who need reproducibility should pass a seeded rng.
                Action selection is ``rng.choice(list(actions))``;
                object selection is deterministic (top-1 by count);
                subject selection is ``rng.choice(top_n_subjects)``.

        Returns:
            The generated sentence as ``"subject action object"``, or
            ``None`` when:

              * the learner is not trained,
              * ``cluster_id`` is not a real cluster (e.g. -1, or an
                id not in ``self.action_clusters``),
              * the cluster has no actions,
              * the chosen action has no objects in
                ``action_object_freq`` (rare — actions with zero
                observations never get clustered),
              * no subject candidate is available (extremely rare —
                only when ``positional_freq`` is empty, which means
                ``is_trained`` is also False).

            Returning ``None`` (rather than raising) lets the CLI
            loop skip degenerate clusters silently.
        """
        if not self.is_trained:
            return None
        if cluster_id not in self.action_clusters:
            return None
        actions = self.action_clusters[cluster_id]
        if not actions:
            return None

        if rng is None:
            import random
            rng = random.Random()

        # 1. Pick an action — uniform over cluster members so every
        #    action gets a chance to surface in the feedback loop.
        action = rng.choice(sorted(actions))

        # 2. Pick the object — highest-count object in
        #    action_object_freq[action]. Ties broken alphabetically
        #    for determinism. The object is the SEMANTIC content the
        #    user evaluates.
        objs = self.action_object_freq.get(action, {})
        if not objs:
            return None
        # max() with a (count, -token) key gives highest count,
        # alphabetically smallest token on ties.
        obj = max(objs.items(), key=lambda kv: (kv[1], -ord(kv[0][0]) if kv[0] else 0))[0]

        # 3. Pick a subject — from tokens with the highest agent-bucket
        #    (bucket 0) counts. We don't track action↔subject
        #    co-occurrence (only action↔object), so this is
        #    corpus-wide. The subject is a grammatical carrier; the
        #    semantic content the user evaluates is the action↔object
        #    pair. Excluding the chosen action and object tokens from
        #    the subject candidate set prevents degenerate sentences
        #    like "makan makan makan" when the same token appears in
        #    multiple buckets.
        #
        #    We also exclude two classes of tokens that are NOT real
        #    subjects but happen to sit at position 0:
        #      (a) ``function_word_candidates`` — statistically
        #          discovered function words (sangat, itu, bukan, ...).
        #      (b) Tokens whose bucket distribution is DOMINATED by
        #          bucket 0 with zero presence at the object bucket
        #          (2 or -1) AND high frequency (>=
        #          ``_SUBJECT_DISCOURSE_MARKER_MIN_FREQ``). These are
        #          discourse markers like "secara", "menurut",
        #          "karena", "begitu" that introduce clauses and never
        #          appear as objects. Real subject nouns (api, manusia,
        #          kucing, ...) also appear as objects somewhere in a
        #          large corpus, so their bucket distribution spans 0
        #          AND 2/-1. Discourse markers don't — and they appear
        #          often (they're grammatical). The frequency floor
        #          excludes rare subject nouns in small corpora (where
        #          a noun might only appear at bucket 0 by chance).
        subject_candidates: List[Tuple[str, int]] = []
        for token, pos_map in self.positional_freq.items():
            if token == action or token == obj:
                continue
            if token in self.function_word_candidates:
                continue
            agent_count = pos_map.get(_AGENT_BUCKET, 0)
            if agent_count <= 0:
                continue
            object_count = (
                pos_map.get(_OBJECT_BUCKET_3, 0)
                + pos_map.get(_OBJECT_BUCKET_N, 0)
            )
            if object_count == 0 and agent_count >= _SUBJECT_DISCOURSE_MARKER_MIN_FREQ:
                # High-frequency token that ONLY appears at bucket 0 —
                # a discourse marker, not a real subject. Skip.
                continue
            subject_candidates.append((token, agent_count))
        if not subject_candidates:
            return None
        # Sort by (-count, token) so the highest-count subject comes
        # first. We pick from the top 5 to add a little variety when
        # the rng is seeded — pure top-1 would make every call
        # produce the same sentence (boring for the user).
        subject_candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        top_n = subject_candidates[:5]
        subject = rng.choice([t for t, _ in top_n])

        return f"{subject} {action} {obj}"

    def label_clusters(
        self,
        mapping: Dict[int, RelationType],
        graph_has_existing_edges: bool = False,
    ) -> None:
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
            graph_has_existing_edges: Set to True when this PCL
                instance is already wired into a
                :class:`TrisynapticCircuit` (or any other component)
                that has **already encoded at least one Episome**
                into an :class:`EngramComplex` graph using this
                learner's current cluster labels. When True, this
                method emits a ``RuntimeWarning`` describing the
                mixed-type-edge risk (see below). When False (the
                default), the call is silent — this is the safe case
                where labelling happens before any edges are encoded.

                **Why this flag exists.** ``TrisynapticCircuit.encode()``
                snapshots ``edge_type`` onto each :class:`Episome`
                and :class:`TypedEdge` at encode time, using the
                PCL's labels *at that moment*. Re-labelling a PCL
                after edges already exist leaves the graph with
                edges whose ``relation_type`` reflects the **old**
                labelling, while new edges (encoded after the
                re-labelling) reflect the **new** labelling. If a
                predicate's cluster label changed between the two
                encodings, the same predicate will have edges with
                **different** ``relation_type`` in the same graph.
                This breaks BA 44's transitivity rules
                (``CAUSAL_CHAIN``, ``CATEGORICAL_TRANSITIVITY``,
                ``FUNCTIONAL_COMPOSITION``), which require
                homogeneous-type chains to fire. See issue #91 for
                the full red-team analysis.

                **The warning is non-blocking** (``RuntimeWarning``,
                not an exception) because there are legitimate
                use cases for re-labelling mid-session (e.g. an
                A/B experiment that flips labels to observe the
                downstream effect on new edges). Researchers who
                know what they're doing can suppress the warning
                with ``warnings.simplefilter('ignore', RuntimeWarning)``
                scoped to the call.

                **The safe path** is to call ``label_clusters()``
                before constructing any component that encodes
                edges (i.e. before ``AGNNCore.__init__`` or before
                the first ``TrisynapticCircuit.encode()`` call).
                In that order, the flag can stay False.
        """
        if graph_has_existing_edges:
            import warnings
            warnings.warn(
                "PositionalClusterLearner.label_clusters() called with "
                "graph_has_existing_edges=True. Existing edges in the "
                "graph retain their original relation_type (snapshotted "
                "at encode time); only new edges will reflect the new "
                "labels. If any predicate's cluster label changed, the "
                "graph will contain mixed-type edges for the same "
                "predicate, which mutes BA 44 transitivity rules on "
                "chains that include those edges. The safe path is to "
                "label clusters BEFORE encoding any edges. See issue #91.",
                RuntimeWarning,
                stacklevel=2,
            )
        for cluster_id, relation_type in mapping.items():
            if cluster_id in self.action_clusters:
                self.cluster_labels[cluster_id] = relation_type

    # ------------------------------------------------------------------
    # Public API: particle cluster inspection + post-hoc naming
    # ------------------------------------------------------------------

    def inspect_particle_clusters(self) -> Dict[int, Dict[str, object]]:
        """Human-readable view of the unnamed particle clusters.

        Returns ``{cluster_id: {"tokens": [...], "label":
        Optional[str], "signature": {...}}}`` — what a human reviews
        before deciding the cluster -> name mapping via
        :meth:`label_particle_clusters`. ``signature`` is the mean
        feature vector across the cluster's members (rounded for
        readability), so a reviewer can see WHY tokens ended up
        together without re-running the math.
        """
        out: Dict[int, Dict[str, object]] = {}
        for cluster_id in sorted(self.particle_clusters.keys()):
            tokens = sorted(self.particle_clusters[cluster_id])
            sigs = [self._compute_particle_signature(t) for t in tokens]
            n = len(sigs) or 1
            mean_sig = tuple(
                round(sum(s[d] for s in sigs) / n, 3) for d in range(4)
            )
            out[cluster_id] = {
                "tokens": tokens,
                "label": self.particle_cluster_labels.get(cluster_id),
                "signature": {
                    "pre_object_3tok_rate": mean_sig[0],
                    "between_first_rate": mean_sig[1],
                    "fine_position_entropy": mean_sig[2],
                    "bucket_entropy": mean_sig[3],
                },
            }
        return out

    def label_particle_clusters(self, mapping: Dict[int, str]) -> None:
        """Assign human-readable names to particle clusters (post-hoc, once).

        Mirrors :meth:`label_clusters` for RelationType. ``mapping``
        is ``{cluster_id: name}`` where ``name`` can be ANY string —
        not constrained to "MODIFIER"/"CONNECTOR". A cluster's
        contents might warrant a name nobody anticipated; the
        architecture does not presuppose what grammar classes exist
        beyond AGENT/ACTION/OBJECT (which are positional, not
        cluster-based — see :meth:`tag_sentence`).

        Idempotent: calling again with a different mapping overwrites
        previous labels. Unknown cluster_ids are silently skipped.
        """
        for cluster_id, name in mapping.items():
            if cluster_id in self.particle_clusters:
                self.particle_cluster_labels[cluster_id] = name

    # ------------------------------------------------------------------
    # Public API: classification
    # ------------------------------------------------------------------

    def get_relation_type_for_action(
        self, action: str,
    ) -> Optional[RelationType]:
        """Resolve the ``RelationType`` for an action token by content.

        Resolves ``action`` → ``cluster_id`` → ``cluster_labels[cluster_id]``,
        returning the label without exposing the unstable cluster_id to
        the caller (see issue #93: cluster IDs are not stable across
        PCL versions; code that introspects by cluster_id directly
        breaks silently on PCL upgrades).

        Args:
            action: The action token to look up. Will be normalized
                via :meth:`_normalize_token` (lower-cased + whitespace-
                collapsed). Multi-word predicates should pass the
                normalized form (e.g. ``"bergantung pada"`` for the
                FUNCTIONAL predicate; the head word ``"bergantung"``
                is also accepted and will be looked up directly).

        Returns:
            The :class:`RelationType` of the labelled cluster that
            contains ``action``, or ``None`` if:

              - the learner is not trained (no clusters discovered), or
              - the action is not tracked by the learner (never
                observed in the training corpus), or
              - the action's cluster_id is ``-1`` (unclustered, below
                ``min_action_observations``), or
              - the action's cluster exists but is not labelled
                (``cluster_labels`` has no entry for it).

            ``None`` always means "fall back to the wrapped
            :class:`SemanticRoleClassifier`" — callers should treat
            ``None`` as "no PCL-primary classification available".

        Why this method exists
        -----------------------
        Pre-this-method, downstream code (e.g.
        :meth:`AGNNCore.apply_feedback`) reached into
        ``classifier.cluster_id_of[action]`` and then
        ``classifier.cluster_labels[cluster_id]`` directly. This is
        fragile because cluster IDs are an **implementation detail**
        of the greedy agglomerative merge in :meth:`_cluster_actions`:

          - They depend on corpus token order.
          - They depend on ``similarity_threshold``.
          - They depend on the presence/absence of anchor-word
            discovery (PR #81 changed this).
          - They depend on the presence/absence of Brown clustering
            for objects (PR #81 changed this).

        Any change to the clustering algorithm shifts cluster IDs.
        Code that hardcodes ``cluster_labels[42]`` to fetch CAUSAL
        will silently inspect the wrong cluster after a PCL upgrade.

        This method hides the cluster_id behind a content-addressed
        lookup: ``action`` → ``cluster_id_of[action]`` →
        ``cluster_labels[cluster_id]``. The caller never sees the ID.

        Backward compatibility
        ----------------------
        The internal fields ``cluster_id_of`` and ``cluster_labels``
        are NOT deprecated — they remain the implementation backing
        this method, and tests that exercise the implementation
        directly (e.g. ``test_positional_cluster_learner.py``'s
        cluster-identity assertions) still need them. The new API is
        the **preferred** way for production code (anything outside
        PCL's own test suite) to look up an action's label.

        Relation to :meth:`classify`
        ----------------------------
        :meth:`classify` takes a full ``text`` sentence, parses SPO,
        checks negation, then looks up the action's cluster label —
        falling back to :class:`SemanticRoleClassifier` on any miss.
        This method takes just an action token and returns only the
        PCL-primary label (or ``None``); it does NOT fall back to
        SRC. Use this method when you want to know "did PCL label
        this action?" without invoking the full fallback chain.

        Example
        -------
        >>> learner = PositionalClusterLearner.load("cluster_learner_state.json")
        >>> learner.get_relation_type_for_action("menyebabkan")
        <RelationType.CAUSAL: 'causal'>
        >>> learner.get_relation_type_for_action("memicu")
        <RelationType.CAUSAL: 'causal'>
        >>> learner.get_relation_type_for_action("upload")  # unclustered loan-word
        None
        >>> learner.get_relation_type_for_action("merawat")  # clustered but unlabelled
        None
        """
        # Defensive: if not trained, there's nothing to look up.
        if not self.is_trained:
            return None
        # Defensive: non-string input (None, int, etc.) returns None
        # rather than raising AttributeError on .lower(). This makes
        # the API robust to callers that pass untyped user input.
        if not isinstance(action, str):
            return None
        # Normalize the input the same way classify() does.
        action_token = self._normalize_token(action)
        if not action_token:
            return None
        # Look up the cluster_id for this action (if tracked at all).
        cluster_id = self.cluster_id_of.get(action_token)
        if cluster_id is None or cluster_id < 0:
            # None = untracked (never observed); -1 = tracked but
            # unclustered (below min_action_observations). Either way,
            # no PCL-primary classification.
            return None
        # Look up the cluster's label (if labelled).
        return self.cluster_labels.get(cluster_id)

    # ------------------------------------------------------------------
    # Public API: object super-cluster inspection (POS-class discovery)
    # ------------------------------------------------------------------

    def get_object_supercluster(
        self, token: str,
    ) -> Optional[int]:
        """Return the Brown super-cluster ID for an object token.

        Resolves ``token`` → ``object_supercluster_id[token]``. The
        super-cluster ID groups object tokens that share similar
        action-context distributions (e.g. ``mamalia``, ``logam``,
        ``reptil`` all belong to the same "taxonomy noun" super-cluster
        because they co-occur with the same set of copulas).

        Args:
            token: The object token to look up. Will be normalized
                via :meth:`_normalize_token` (lower-cased +
                whitespace-collapsed).

        Returns:
            The super-cluster ID (a non-negative int), or ``None`` if:

              - the learner is not trained, or
              - the token is not in the object vocabulary (never
                observed as an object of any action), or
              - the token was observed but didn't merge into any
                super-cluster (singleton — assigned its own ID, but
                this is still a non-None return).

            ``None`` always means "not in the object vocabulary".

        Why this method exists
        -----------------------
        The Brown super-clustering of object vocabulary was previously
        an internal helper for action clustering (see
        :meth:`_cluster_object_vocabulary`). Exposing it as a public
        API lets downstream code query "what kind of object is this?"
        without reaching into the internal ``object_supercluster_id``
        dict — same zero-bias principle as
        :meth:`get_relation_type_for_action` (issue #93).
        """
        if not self.is_trained:
            return None
        if not isinstance(token, str):
            return None
        token = self._normalize_token(token)
        if not token:
            return None
        sc_id = self.object_supercluster_id.get(token)
        return sc_id  # None if not in vocabulary

    def inspect_object_superclusters(self) -> Dict[int, List[str]]:
        """Return all Brown super-clusters as ``{sc_id: [tokens]}``.

        Convenience wrapper around the internal
        ``object_superclusters`` dict that returns a plain
        ``{int: List[str]}`` (sorted) instead of ``{int: Set[str]}``.
        Useful for debugging, logging, and corpus inspection — e.g.
        checking whether taxonomy nouns (``mamalia``, ``logam``) and
        property adjectives (``dingin``, ``asam``) ended up in
        different super-clusters.

        Returns:
            ``{supercluster_id: sorted_list_of_object_tokens}``. Empty
            dict if the learner is untrained or no objects were
            clustered.
        """
        return {
            sc_id: sorted(objs)
            for sc_id, objs in self.object_superclusters.items()
        }

    # ------------------------------------------------------------------
    # Public API: unified POS tagging (POS-class discovery)
    # ------------------------------------------------------------------

    def tag_sentence(self, text: str) -> List[Tuple[str, str]]:
        """Tag every token in ``text`` with a POS class.

        Returns a list of ``(token, pos_class)`` pairs for every token
        in the input sentence. ``pos_class`` is one of:

            - ``"AGENT"``: the subject (position 0 in SVO).
            - ``"ACTION"``: the predicate (position 1 in SVO, or the
              action-bucket position in >3-token sentences). Only
              assigned to tokens that are real actions (in
              ``cluster_id_of`` or ``action_object_freq``, or with
              verb morphology).
            - ``"OBJECT"``: the object (last position in SVO).
            - ``"MODIFIER"``: an adverb / intensifier (in
              ``modifier_tokens`` — discovered via the pre-object
              positional signal).
            - ``"CONNECTOR"``: a preposition / complementizer (in
              ``connector_tokens`` — discovered via the between-first
              positional signal).
            - ``"UNKNOWN"``: the token doesn't have enough data to be
              classified into any of the above categories.

        The tagging is **purely positional + statistical** — no
        hardcoded word lists. Each token's POS class is determined by:

          1. Parse SPO positionally via :meth:`spo` to identify the
             AGENT (subject), ACTION (predicate), and OBJECT slots.
          2. If the token at the ACTION slot is a real action (in
             ``action_object_freq`` or ``cluster_id_of``, or has verb
             morphology), tag it ``ACTION``. Otherwise it falls
             through to step 3-5.
          3. If the token is in ``modifier_tokens``, tag it
             ``MODIFIER``.
          4. If the token is in ``connector_tokens``, tag it
             ``CONNECTOR``.
          5. Otherwise tag it ``UNKNOWN``.

        Tokens at the AGENT and OBJECT slots are always tagged
        ``AGENT`` and ``OBJECT`` respectively, regardless of their
        membership in modifier/connector sets (positional role beats
        grammatical class — a noun is still a noun even if it happens
        to be in the modifier set by corpus noise).

        Args:
            text: The sentence to tag. Tokenized via :meth:`_tokenize`
                (lower-cased + punctuation stripped + whitespace
                collapsed).

        Returns:
            List of ``(token, pos_class)`` pairs. Empty list for empty
            / unparseable input. For sentences shorter than 3 tokens,
            positional SVO parsing is ambiguous, so the method falls
            back to per-token grammar-class lookup (MODIFIER /
            CONNECTOR / UNKNOWN) without AGENT/ACTION/OBJECT
            assignment.

        Example
        -------
        >>> learner = PositionalClusterLearner.load("cluster_learner_state.json")
        >>> learner.tag_sentence("es sangat dingin")
        [("es", "AGENT"), ("sangat", "MODIFIER"), ("dingin", "OBJECT")]
        >>> learner.tag_sentence("kucing berbeda dari reptil")
        [("kucing", "AGENT"), ("berbeda", "ACTION"), ("dari", "CONNECTOR"), ("reptil", "OBJECT")]
        """
        if not self.is_trained:
            # Untrained — can't do positional SVO parsing. Fall back
            # to per-token grammar-class lookup (all UNKNOWN if no
            # modifier/connector sets are populated).
            tokens = self._tokenize(text)
            return [(t, self._lookup_grammar_class(t)) for t in tokens]

        tokens = self._tokenize(text)
        if not tokens:
            return []

        n = len(tokens)

        # For sentences with < 3 tokens, positional SVO is ambiguous.
        # Fall back to per-token grammar-class lookup.
        if n < 3:
            return [(t, self._lookup_grammar_class(t)) for t in tokens]

        # Parse SVO positionally.
        try:
            spo = self.spo(text)
        except Exception:
            # spo() failed — fall back to per-token lookup.
            return [(t, self._lookup_grammar_class(t)) for t in tokens]

        subject = self._normalize_token(spo.subject) if spo.subject else ""
        # spo() returns predicate as a phrase for >3-token sentences.
        # The head verb is the first token of the predicate phrase.
        predicate_head = ""
        if spo.predicate:
            predicate_head = self._normalize_token(spo.predicate.split()[0])
        obj = self._normalize_token(spo.object) if spo.object else ""

        # Build the tag list. Start with UNKNOWN for everything, then
        # fill in AGENT/ACTION/OBJECT/MODIFIER/CONNECTOR.
        tags: List[str] = ["UNKNOWN"] * n

        # Identify AGENT (position 0) and OBJECT (last position) by
        # matching the SPO parse against the token list.
        if subject and tokens[0] == subject:
            tags[0] = "AGENT"
        if obj and tokens[n - 1] == obj:
            tags[n - 1] = "OBJECT"

        # Identify ACTION (position 1, or the action-bucket position).
        # The predicate head verb should be at index 1 in 3-token
        # sentences, or at index 1 in >3-token sentences (per
        # _compute_buckets: [0] + [1]*(n-2) + [-1]).
        if predicate_head and n >= 2 and tokens[1] == predicate_head:
            # Check if this token is a real action. It's a real action
            # if it's in action_object_freq / cluster_id_of, or if it
            # has verb morphology. Otherwise it might be a modifier
            # sitting at the action slot (e.g. 'sangat' in
            # "es sangat dingin").
            #
            # Note: we do NOT check action_bucket_anchors here. A
            # token can be an action_bucket_anchor (sits at the action
            # bucket frequently) AND a modifier — 'sangat' is both.
            # The modifier classification takes priority for tokens
            # that are NOT in action_object_freq (i.e. were excluded
            # from action extraction because they're function words).
            is_real_action = (
                predicate_head in self.action_object_freq
                or predicate_head in self.cluster_id_of
                or self._looks_like_verb(predicate_head)
            )
            if is_real_action:
                tags[1] = "ACTION"
            # If not a real action, leave it as UNKNOWN for now —
            # the grammar-class lookup below will classify it as
            # MODIFIER if applicable.

        # Fill in MODIFIER and CONNECTOR for tokens that aren't
        # already tagged AGENT/ACTION/OBJECT.
        for i, tok in enumerate(tokens):
            if tags[i] != "UNKNOWN":
                continue
            gc = self._lookup_grammar_class(tok)
            tags[i] = gc

        return list(zip(tokens, tags))

    def _lookup_grammar_class(self, token: str) -> str:
        """Return the grammar class for a token.

        Sole mechanism: look up ``particle_cluster_id_of[token]`` ->
        ``particle_cluster_labels`` (the zero-bias, post-hoc-named
        particle cluster — see the "PARTICLE clustering" module
        section). Content-addressed (issue #93 pattern): the
        cluster_id is never exposed, only the assigned name.

        Deliberately does NOT fall back to the legacy direct-naming
        sets (``modifier_tokens`` / ``connector_tokens``) — those sets
        bake the name "MODIFIER"/"CONNECTOR" into the detection
        function itself (a human decided the answer, then tuned a
        threshold until specific known tokens matched it), which is
        the same pattern PR #69 was rejected for. Falling back to them
        here would mean every unlabelled particle cluster silently
        reports the OLD pre-named answer instead of the honest
        "UNKNOWN" — defeating the purpose of the two-stage discovery.

        A token reports ``"UNKNOWN"`` until a human has reviewed
        :meth:`inspect_particle_clusters` and called
        :meth:`label_particle_clusters` for its cluster. This is the
        correct default: "not yet classified" is a more honest signal
        than "classified using a name nobody verified against this
        token's actual cluster membership".

        Note: this method does NOT identify AGENT/ACTION/OBJECT —
        those are positional roles determined by :meth:`spo`, not by
        cluster membership. This method only identifies the grammar-
        class overlay for tokens whose positional role is ambiguous
        or non-structural.
        """
        cluster_id = self.particle_cluster_id_of.get(token)
        if cluster_id is not None and cluster_id >= 0:
            label = self.particle_cluster_labels.get(cluster_id)
            if label is not None:
                return label
        return "UNKNOWN"

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
              "similarity_threshold": 0.13,
              "min_action_observations": 2,
              "positional_freq":     {"makan": {"1": 10}, ...},
              "fine_positional_freq":{"sangat": {"1": 11, "2": 10, ...}, ...},
              "action_object_freq":  {"menyebabkan": {"panas": 2, ...}, ...},
              "cluster_id_of":       {"makan": 0, "minum": 0, ...},
              "action_clusters":     {"0": ["makan", "minum", ...], ...},
              "cluster_labels":      {"0": "CAUSAL", ...},    # may be empty
              "action_connector_signature": {"berbeda": true, "adalah": false, ...},
              "connector_tokens":    ["dari", "dengan", "sebagai", ...],
              "function_word_candidates":   ["sangat", "itu", "bukan", ...],
              "action_bucket_anchors":      ["adalah", "merupakan", ...],
              "object_supercluster_id":     {"mamalia": 0, "logam": 0, ...},
              "object_superclusters":       {"0": ["mamalia", "logam", ...], ...},
              "pre_object_3tok_freq":       {"sangat": 30, "begitu": 12, ...},
              "pre_object_long_freq":       {"dari": 51, "dengan": 81, ...},
              "modifier_tokens":            ["sangat", "begitu", ...]
            }

        The ``fine_positional_freq``, ``function_word_candidates``,
        ``action_bucket_anchors``, ``object_supercluster_id``,
        ``object_superclusters``, ``pre_object_3tok_freq``,
        ``pre_object_long_freq``, and ``modifier_tokens`` fields are persisted so a loaded
        learner reproduces the same zero-bias anchor-word discovery,
        Brown-clustering contract, and MODIFIER discovery. Older save
        files (pre-anchor / pre-Brown / pre-MODIFIER fix) lack these
        fields and load() backfills them as empty / re-derives them
        from positional_freq — backward compatible.

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
            "fine_positional_freq": {
                tok: {str(p): c for p, c in fine_map.items()}
                for tok, fine_map in self.fine_positional_freq.items()
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
            "function_word_candidates": sorted(self.function_word_candidates),
            "action_bucket_anchors": sorted(self.action_bucket_anchors),
            "object_supercluster_id": dict(self.object_supercluster_id),
            "object_superclusters": {
                str(sc_id): sorted(objs)
                for sc_id, objs in self.object_superclusters.items()
            },
            "pre_object_3tok_freq": dict(self.pre_object_3tok_freq),
            "pre_object_long_freq": dict(self.pre_object_long_freq),
            "modifier_tokens": sorted(self.modifier_tokens),
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

        .. warning::
            **Defensive backfill branches — DO NOT remove.** Every
            ``raw.get(<field>, <default>)`` call below that
            backfills an empty container when the field is absent is
            **intentional insurance against state-file schema drift**.
            These branches protect three real-world scenarios:

              1. **Hand-edited state files.** A user (or developer
                 debugging) may hand-edit the JSON and accidentally
                 drop a field. The backfill lets the file still load
                 with the pre-field behaviour rather than crashing.
              2. **State files from older releases.** A state file
                 saved before a field was introduced (e.g.
                 ``fine_positional_freq`` was added in the
                 anchor-word fix, ``object_supercluster_id`` was
                 added in the Brown-clustering fix) must still load
                 on the current code. The backfill re-derives or
                 defaults the missing field.
              3. **State files from future releases.** A state file
                 saved by a newer version that drops a deprecated
                 field must still load on the current code (e.g. when
                 rolling back a release).

            On the canonical shipped state file
            (``AGNN/data/cluster_learner_state.json``) — which is
            regenerated by the current ``bootstrap_classifier`` —
            none of these branches fire; the file always contains
            every field. They fire only on out-of-band inputs. See
            ``AGNN/docs/dead-code-audit.md`` §3.6 for the original
            "keep as cheap insurance" decision.

        .. warning::
            **Post-encode mutation risk (issue #91).** Loading a
            state file whose ``cluster_labels`` differ from the
            labels used at encode time is functionally equivalent
            to calling :meth:`label_clusters` on a PCL that's
            already wired into a graph with existing edges. Existing
            :class:`TypedEdge` instances retain the ``relation_type``
            that was snapshotted at encode time (using the *old*
            labels); only edges encoded after the load will reflect
            the *new* labels. If a predicate's cluster label
            changed between the two states, the graph will contain
            mixed-type edges for the same predicate, which mutes
            BA 44 transitivity rules on chains that include those
            edges.

            **Production scenario:** deploy v1 with state file v1 →
            user learns N facts → deploy v2 with retrained state
            file v2 (different ``cluster_labels``) → user learns M
            more facts. The graph now has mixed ``relation_type``
            for any predicate whose label shifted between v1 and v2.

            **Mitigation:** version-stamp your state files and
            refuse to load a state file whose version differs from
            the one used to encode the existing graph. Or: rebuild
            the graph from scratch after every PCL state change.
            See :meth:`label_clusters` for the in-session equivalent
            of this warning (with the ``graph_has_existing_edges``
            flag).
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

        # fine_positional_freq: {token: {pos_str: count}} -> {token: {int: int}}
        # Backward-compat: older save files (pre-anchor-word fix) lack
        # this field. We re-derive it from positional_freq where possible,
        # but the lossy coarse-bucket scheme means the derived version
        # is approximate (only buckets 0/1/2/-1, not raw indices). The
        # learner will still work; re-calling train() will rebuild the
        # accurate fine_positional_freq.
        for tok, fine_map in raw.get("fine_positional_freq", {}).items():
            if not isinstance(tok, str) or not isinstance(fine_map, dict):
                continue
            learner.fine_positional_freq[tok] = {
                int(p): int(c)
                for p, c in fine_map.items()
                if isinstance(p, str) and isinstance(c, (int, float))
            }

        # function_word_candidates: list[str] -> set[str]
        # Backward-compat: older save files lack this field. Empty set
        # is the correct backfill — _extract_action_object treats the
        # empty set as "no function words discovered", which preserves
        # the pre-fix behaviour of accepting all position-1 tokens.
        for tok in raw.get("function_word_candidates", []):
            if isinstance(tok, str):
                learner.function_word_candidates.add(tok)

        # action_bucket_anchors: list[str] -> set[str]
        # Same backward-compat note as above.
        for tok in raw.get("action_bucket_anchors", []):
            if isinstance(tok, str):
                learner.action_bucket_anchors.add(tok)

        # object_supercluster_id: {obj: sc_id} -> {str: int}
        # Backward-compat: older save files lack this field. Empty dict
        # is the correct backfill — _cluster_actions falls back to
        # hash(object_token) for super-cluster ids when this dict is
        # empty, which preserves the pre-Brown-fix behaviour of
        # clustering by literal object tokens.
        for obj, sc_id in raw.get("object_supercluster_id", {}).items():
            if isinstance(obj, str) and isinstance(sc_id, int):
                learner.object_supercluster_id[obj] = sc_id

        # object_superclusters: {sc_id_str: [objs]} -> {int: set}
        for sc_id_str, objs in raw.get("object_superclusters", {}).items():
            if not isinstance(sc_id_str, str) or not isinstance(objs, list):
                continue
            try:
                sc_id_int = int(sc_id_str)
            except ValueError:
                continue
            learner.object_superclusters[sc_id_int] = {
                o for o in objs if isinstance(o, str)
            }

        # pre_object_3tok_freq: {token: count} -> {str: int}
        # Backward-compat: older save files (pre-MODIFIER-discovery)
        # lack this field. Empty dict is the correct backfill —
        # modifier_tokens will also be empty, so tag_sentence() will
        # classify any unknown mid-sentence token as UNKNOWN (which is
        # the safe default for a learner loaded from a stale state
        # file). Re-running train() on the canonical corpus rebuilds
        # all three fields.
        for tok, count in raw.get("pre_object_3tok_freq", {}).items():
            if isinstance(tok, str) and isinstance(count, (int, float)):
                learner.pre_object_3tok_freq[tok] = int(count)

        # pre_object_long_freq: {token: count} -> {str: int}
        # Same backward-compat note as above.
        for tok, count in raw.get("pre_object_long_freq", {}).items():
            if isinstance(tok, str) and isinstance(count, (int, float)):
                learner.pre_object_long_freq[tok] = int(count)

        # modifier_tokens: list[str] -> set[str]
        # Same backward-compat note as above.
        for tok in raw.get("modifier_tokens", []):
            if isinstance(tok, str):
                learner.modifier_tokens.add(tok)

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

    def _looks_like_verb(self, token: str) -> bool:
        """Heuristic: does this token look like an Indonesian verb?

        Indonesian verbs are highly morphologically regular. The vast
        majority start with me-, ber-, di-, or ter- (and their
        allomorphs meng-/meny-/mem-/men-, bel-, diper-, etc.). This
        coarse morphological signal lets us distinguish verbs from
        nouns when positional parsing alone would be ambiguous - e.g.
        in multi-word subjects like "ahli gizi menyarankan" where
        "gizi" sits at position 1 (noun, part of compound subject)
        but "menyarankan" at position 2 is the real verb.

        This is a MORPHOLOGICAL signal (word form), not a SEMANTIC
        one (word meaning). It does NOT violate the zero-bias
        principle. The old ``_COPULAS`` whitelist (``adalah``,
        ``merupakan``, ``ialah``, ``yaitu``, ``yakni``) was removed
        because it was a meaning-based list; copulas are now
        recognised via ``self.action_bucket_anchors`` (statistical
        discovery from positional concentration) instead of via this
        morphological heuristic.

        Conservative by design:
          - All prefixes are 3+ characters to avoid false positives
            ("di" alone is a preposition, "me" alone matches "merah").
          - We accept some false positives ("beras" = rice, "ternak" =
            livestock) because they're rare in the action slot and the
            cost of false negatives (breaking the multi-word subject
            fix) is much higher.
          - We accept some false negatives ("makan", "minum", "ambil",
            "adalah", "merupakan" don't carry these prefixes) because
            the caller (``_extract_action_object``) recognises action
            bucket anchors as a fallback for non-morphological verbs.
        """
        if not token or len(token) < 3:
            return False
        return token.startswith(_VERB_PREFIXES)

    def _extract_action_object(
        self, tokens: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract (action_token, object_token) by CURRENT position.

        For 3-token sentences: action=tokens[1], object=tokens[2].
        For >3-token sentences: action=first verb-looking token (must
            start with me-/ber-/diper-/ter-, OR be a discovered
            action_bucket_anchor) before the object slot;
            object=tokens[-1]. If no such token is found, the
            sentence is skipped (returns ``(None, None)``).
        For <3-token sentences: ``(None, None)`` - no SVO structure.

        This is the polysemy fix in action: the action and object are
        determined by WHERE they sit in *this* sentence, not by any
        global cluster membership. The same token "ayam" can be the
        object of "manusia potong ayam" and the agent of
        "ayam mencari pakan" - both are recorded correctly because
        positional_freq is a *soft* count.

        Zero-bias function-word exclusion (replaces Bug 1 stoplist):
        Function words in ``self.function_word_candidates``
        (statistically discovered from positional entropy - see
        ``_compute_anchor_words``) are skipped from the action slot.
        This prevents garbage clusters like ``{'actions': ['sangat'],
        'top_objects': ['asin', 'dingin']}`` that previously formed
        when state+adjective sentences ("es itu sangat dingin") had
        their intensifier captured as the action. The set is
        discovered purely from positional statistics - no hardcoded
        word list.

        MODIFIER exclusion (POS-class discovery):
        Tokens in ``self.modifier_tokens`` (statistically discovered
        from pre-object positional signal - see
        ``_compute_modifiers``) are also skipped from the action
        slot. This is the same exclusion as function_word_candidates
        but for a different grammar class: modifiers like 'sangat'
        that sit at the action slot in 3-token "state + adj"
        sentences but are NOT real actions. Without this exclusion,
        'sangat' would be extracted as the action in "es sangat
        dingin", polluting action_object_freq and preventing it from
        being classified as a MODIFIER.

        Zero-bias copula recognition (replaces ``_COPULAS`` whitelist):
        Copulas like ``adalah`` and ``merupakan`` lack the me-/ber-/
        diper-/ter- prefixes that ``_looks_like_verb`` detects, so
        they would be skipped in >3-token sentences under the pure
        morphological heuristic. Instead, ``self.action_bucket_anchors``
        (statistically discovered from positional concentration at
        the action bucket - see ``_compute_anchor_words``) lets any
        token that concentrates at the action bucket be recognised
        as a valid action. ``adalah`` (bucket_freq={1: 69},
        bucket_nh=0.0) and ``merupakan`` (bucket_freq={1: 103},
        bucket_nh=0.0) emerge as anchors automatically.

        Multi-word subject fix - verb-prefix OR anchor requirement:
        Sentences with >3 tokens may have a multi-word subject like
        "ahli gizi" or "dokter kulit" occupying positions 0-1, which
        means position 1 is a noun (not the action). To avoid
        capturing that noun as the action, we require the action
        candidate to either (a) start with me-/ber-/diper-/ter- or
        (b) be a discovered action_bucket_anchor. If no such token
        exists before the object slot, the sentence is skipped.

        For 3-token sentences (classic SVO with no room for multi-word
        subjects), the positional parse is unambiguous. We accept
        position 1 as the action when:
          - it is NOT a function word candidate; AND
          - it has verb morphology OR its frequency at the action
            bucket is >= ``_3_TOKEN_MIN_ACTION_FREQ`` (the floor
            that excludes one-off function words in small test
            corpora while admitting recurring irregular verbs like
            'makan').

        When the action is also the last token (no candidate remains
        after it, as in pure state+adjective patterns like
        "es itu dingin"), the sentence has no real object and is
        skipped from ``action_object_freq`` by returning
        ``(None, None)``.
        """
        if len(tokens) < 3:
            return None, None

        if len(tokens) == 3:
            # 3-token SVO: no room for a multi-word subject, so the
            # positional parse is unambiguous. Take position 1 as the
            # action IF:
            #   1. It is NOT a function word candidate (statistically
            #      discovered via positional entropy - see
            #      _compute_anchor_words).
            #   2. It has appeared at the action bucket at least
            #      _3_TOKEN_MIN_ACTION_FREQ times across the corpus.
            #
            # The frequency floor is the key filter: it excludes
            # one-off function words in synthetic test corpora (e.g.,
            # 'memang' at freq 1, which false-positives the verb-
            # morphology heuristic because it starts with 'mem-')
            # while admitting recurring verbs (e.g., 'makan' at
            # freq 2). In real corpora, function word candidates
            # are already excluded by the entropy-based discovery,
            # and the frequency floor is a backstop for small corpora
            # where statistical discovery doesn't have enough data.
            candidate = tokens[1]
            if candidate in self.function_word_candidates:
                return None, None
            # MODIFIER exclusion: modifiers like 'sangat' sit at the
            # action slot in 3-token "state + adj" sentences but are
            # NOT real actions. Excluding them here prevents pollution
            # of action_object_freq.
            if candidate in self.modifier_tokens:
                return None, None
            action_bucket_count = self.positional_freq.get(
                candidate, {}
            ).get(_ACTION_BUCKET, 0)
            if action_bucket_count < _3_TOKEN_MIN_ACTION_FREQ:
                return None, None
            return candidate, tokens[-1]

        # >3-token sentence: potential multi-word subject. Require the
        # action candidate to have verb morphology OR be a discovered
        # action_bucket_anchor. This prevents nouns like "gizi" (in
        # "ahli gizi menyarankan diet") from being captured as the
        # action when they're really part of the compound subject.
        action_idx = None
        for i in range(1, len(tokens) - 1):
            candidate = tokens[i]
            # Function words are always excluded.
            if candidate in self.function_word_candidates:
                continue
            # MODIFIERs are also excluded (same rationale as 3-token).
            if candidate in self.modifier_tokens:
                continue
            # Verb morphology OR action bucket anchor.
            if self._looks_like_verb(candidate):
                action_idx = i
                break
            if candidate in self.action_bucket_anchors:
                action_idx = i
                break
        if action_idx is None:
            # No verb-looking or anchor token before the object slot.
            # Skip this sentence to avoid noun-as-action garbage. The
            # cost is losing sentences whose verb is an irregular root
            # (e.g. "makan", "minum") in a >3-token sentence, but
            # that's a small fraction of the corpus and the gain in
            # cluster quality (no garbage noun-actions) is much larger.
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
