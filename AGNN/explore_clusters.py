#!/usr/bin/env python3
"""Exploration sandbox for PositionalClusterLearner.

Separate from the production corpus/state (DEFAULT_CORPUS_PATHS,
cluster_learner_state.json) so large-volume, less-curated text can be
fed in to observe how AGNN's clustering evolves — WITHOUT touching the
sprint-reviewed, test-covered production pipeline.

State lives in AGNN/data/exploration_state.json (learner.save() format,
same as production — issue #92 contract). A companion snapshot file
(exploration_snapshot_prev.json) holds the cluster membership right
before the most recent feed(), used by `diff` to report what changed.

Usage (run from AGNN/):
    python explore_clusters.py status
    python explore_clusters.py feed <path-to-text-file>
    python explore_clusters.py feed -          # read from stdin
    python explore_clusters.py feed-many <file1> <file2> ... <fileN>
                                                # read all concurrently,
                                                # train() ONCE on the merge
    python explore_clusters.py diff
    python explore_clusters.py query "<kalimat>"
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neocortex.bootstrap_classifier import build_labelled_cluster_learner
from neocortex.positional_cluster_learner import PositionalClusterLearner

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_STATE_PATH = os.path.join(_DATA_DIR, "exploration_state.json")
_SNAPSHOT_PATH = os.path.join(_DATA_DIR, "exploration_snapshot_prev.json")
_LOG_PATH = os.path.join(_DATA_DIR, "exploration_feed_log.txt")


def _load_or_init() -> PositionalClusterLearner:
    """Load the exploration state if it exists, else start from the
    canonical, sprint-reviewed production corpus (same as
    build_labelled_cluster_learner()'s default).
    """
    if os.path.exists(_STATE_PATH):
        return PositionalClusterLearner.load(_STATE_PATH)
    print("No exploration state found — bootstrapping from the canonical "
          "production corpus as the starting point.")
    return build_labelled_cluster_learner()


def _snapshot(learner: PositionalClusterLearner) -> Dict[str, Dict[str, List[str]]]:
    """{"action": {cid_str: members}, "particle": {cid_str: members}}.

    Both axes are snapshotted because both are full-retrain outputs of
    train() and both can shift on any feed() call — `diff` was
    previously blind to particle_clusters (only read action_clusters),
    which meant a real movement (e.g. a token leaving a particle
    cluster after corpus expansion) was invisible to the CLI even
    though the underlying state had genuinely changed.
    """
    return {
        "action": {
            str(cid): sorted(members)
            for cid, members in learner.action_clusters.items()
        },
        "particle": {
            str(cid): sorted(members)
            for cid, members in learner.particle_clusters.items()
        },
    }


def _best_match(
    members: Set[str], other_clusters: Dict[str, List[str]],
) -> tuple:
    """Find the other-side cluster with the highest Jaccard overlap.

    Returns (best_cid_or_None, jaccard_score). Used to track a
    cluster's identity across a retrain even though cluster_ids are
    just enumeration order and can shift — same "match by content, not
    by id" principle bootstrap_classifier.py already uses for
    RelationType labelling.
    """
    best_cid = None
    best_score = 0.0
    for cid, other_members in other_clusters.items():
        other_set = set(other_members)
        if not other_set and not members:
            continue
        union = members | other_set
        if not union:
            continue
        score = len(members & other_set) / len(union)
        if score > best_score:
            best_score = score
            best_cid = cid
    return best_cid, best_score


def cmd_status() -> None:
    learner = _load_or_init()
    n_tokens = len(learner.positional_freq)
    n_actions = len(learner.action_object_freq)
    n_clusters = len(learner.action_clusters)
    n_particle_clusters = len(learner.particle_clusters)
    print(f"Exploration state: {_STATE_PATH}")
    print(f"  Total distinct tokens seen : {n_tokens}")
    print(f"  Actions in action_object_freq : {n_actions}")
    print(f"  Action clusters : {n_clusters}")
    print(f"  Particle clusters : {n_particle_clusters}")
    print(f"  Coordinator clusters marked : {sorted(learner.coordinator_cluster_ids)}")
    if os.path.exists(_LOG_PATH):
        with open(_LOG_PATH, encoding="utf-8") as f:
            batches = f.read().count("=== BATCH")
        print(f"  Feed batches so far : {batches}")


def _read_lines(source: str) -> List[str]:
    if source == "-":
        text = sys.stdin.read()
    else:
        with open(source, encoding="utf-8") as f:
            text = f.read()
    return [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _fetch_sources_concurrently(sources: List[str]) -> Dict[str, List[str]]:
    """Read N source files CONCURRENTLY (I/O-bound — threads are safe
    here, no shared mutable state between reads). Returns
    {source: lines}, preserving input order on the caller side.

    This is the part of the pipeline that is SAFE to parallelize:
    independent file reads, no writes, no shared learner state. The
    write side (train() + save()) below this is run sequentially on
    purpose — see cmd_feed_many's docstring for why.
    """
    results: Dict[str, List[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        future_to_source = {
            pool.submit(_read_lines, src): src for src in sources
        }
        for future in concurrent.futures.as_completed(future_to_source):
            src = future_to_source[future]
            results[src] = future.result()
    return results


# Same content-match set bootstrap_classifier.py uses
# (EXPECTED_COORDINATOR_TOKENS) — re-derived here rather than imported
# because explore_clusters.py is deliberately decoupled from the
# production module's labelling contract (see module docstring).
_EXPECTED_COORDINATOR_TOKENS = {"dan", "atau"}


def _mark_coordinator_clusters(learner: PositionalClusterLearner) -> None:
    """Auto-mark clause-coordinator clusters after every train() call.

    Gap found (Round 28): every retrain here left
    ``coordinator_cluster_ids`` empty, because nothing in this script
    ever called ``mark_clause_coordinator_clusters()`` — unlike
    ``bootstrap_classifier.py``, which does this automatically. Effect:
    ``dan``/``atau`` got wrongly tagged ACTION by ``tag_sentence()`` in
    every exploration query (verified: "dan" had
    ``_is_action_token('dan') == True`` here vs ``False`` in
    production). Same content-match pattern as production — mark
    whichever cluster(s) "dan"/"atau" landed in this retrain. Missing
    coordinators are non-fatal (same graceful-degradation contract as
    production), and this never touches PCL's own clustering logic.
    """
    coordinator_cluster_ids = {
        learner.cluster_id_of[token]
        for token in _EXPECTED_COORDINATOR_TOKENS
        if token in learner.cluster_id_of
    }
    if coordinator_cluster_ids:
        learner.mark_clause_coordinator_clusters(coordinator_cluster_ids)


def _commit_batch(lines: List[str], batch_label: str) -> None:
    """Shared commit logic for both cmd_feed and cmd_feed_many:
    snapshot -> train() -> save() -> append to the feed log.

    train()/save() are NOT parallelized and never will be — every
    feed() call is a FULL RETRAIN over the entire accumulated corpus
    (Brown clustering + Q/K/V re-cluster ALL actions seen so far, not
    just the new lines), so running two of these concurrently would
    have both processes load the SAME state, mutate independent
    in-memory copies, and whichever save() finishes last silently
    discards the other's work (a classic lost-update race). The real
    lever for speed is reducing the NUMBER of train() calls — feed
    many sources in one call instead of one-at-a-time — which is
    exactly what cmd_feed_many does by merging concurrently-read
    sources into a single batch before this function ever runs.
    """
    if not lines:
        print("No usable lines found in input (empty or all comments).")
        return

    learner = _load_or_init()

    before = _snapshot(learner)
    with open(_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(before, f, ensure_ascii=False, indent=2)

    print(f"Feeding {len(lines)} new lines ({batch_label}) — accumulates "
          f"on top of existing state...")
    learner.train(lines)
    _mark_coordinator_clusters(learner)
    learner.save(_STATE_PATH)

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== BATCH ({len(lines)} lines, {batch_label}) ===\n")
        for ln in lines:
            f.write(ln + "\n")

    print(f"Saved exploration state to {_STATE_PATH}")
    print(f"Run `python explore_clusters.py diff` to see what changed.")


def cmd_feed(source: str) -> None:
    _commit_batch(_read_lines(source), batch_label=f"source={source}")


def cmd_feed_many(sources: List[str]) -> None:
    """Feed MULTIPLE sources in one go: read them all CONCURRENTLY
    (parallel I/O, safe — independent file reads), merge into a
    single combined batch, then commit with exactly ONE train() call.

    This is the formalized version of what every multi-article batch
    in this exploration session did manually (concatenate N articles
    into one file, then `feed` once) — now an explicit, reusable
    feature instead of an ad-hoc Write-tool step each time.

    Why this is the correct place to parallelize, and why feeding
    each source with its own train() call would NOT be faster: see
    _commit_batch's docstring on the full-retrain cost. N sequential
    `feed` calls pay the full-corpus reclustering cost N times; one
    `feed-many` call pays it ONCE for the same total amount of new
    data.
    """
    print(f"Reading {len(sources)} sources concurrently...")
    results = _fetch_sources_concurrently(sources)
    merged: List[str] = []
    for src in sources:  # preserve caller-specified order
        merged.extend(results.get(src, []))
    print(f"Merged {len(merged)} total lines from {len(sources)} sources.")
    _commit_batch(merged, batch_label=f"{len(sources)} sources merged")


def _print_cluster_diff(
    before: Dict[str, List[str]], after: Dict[str, List[str]], label: str,
) -> None:
    """Print the before->after diff for one cluster axis (action or
    particle). Extracted so both axes share identical diff logic —
    see cmd_diff's docstring for why both need to run.
    """
    before_sets = {cid: set(members) for cid, members in before.items()}
    after_sets = {cid: set(members) for cid, members in after.items()}

    print("=" * 70)
    print(f"{label} CLUSTER DIFF: before -> after most recent feed()")
    print("=" * 70)

    matched_before = set()
    matched_after = set()

    print("\n--- Stable / evolved clusters (matched by token overlap) ---")
    for after_cid, after_members in sorted(after_sets.items(), key=lambda x: -len(x[1])):
        best_before_cid, score = _best_match(after_members, before_sets)
        if best_before_cid is None or score == 0.0:
            continue
        before_members = before_sets[best_before_cid]
        added = after_members - before_members
        removed = before_members - after_members
        # Mark matched BEFORE the "skip if identical" check below — a
        # cluster that didn't change at all is still a real match (its
        # content survived the retrain unchanged), not an absence.
        # Bug found via the particle-axis test: identical clusters were
        # `continue`-d before being marked, so the NEW/DISAPPEARED
        # sections below incorrectly reported every unchanged cluster
        # as both newly created AND disappeared.
        matched_before.add(best_before_cid)
        matched_after.add(after_cid)
        if not added and not removed:
            continue  # identical, nothing to print beyond the match
        print(f"\ncluster {best_before_cid} -> {after_cid} (overlap={score:.2f})")
        if added:
            print(f"  + gained: {sorted(added)}")
        if removed:
            print(f"  - lost:   {sorted(removed)}")

    print("\n--- NEW clusters (no meaningful match in 'before') ---")
    new_count = 0
    for after_cid, after_members in after_sets.items():
        if after_cid in matched_after:
            continue
        new_count += 1
        print(f"  cluster {after_cid}: {sorted(after_members)}")
    if new_count == 0:
        print("  (none)")

    print("\n--- DISAPPEARED clusters (no meaningful match in 'after') ---")
    gone_count = 0
    for before_cid, before_members in before_sets.items():
        if before_cid in matched_before:
            continue
        gone_count += 1
        print(f"  cluster {before_cid}: {sorted(before_members)}")
    if gone_count == 0:
        print("  (none)")

    print(f"\nSummary: {len(before_sets)} clusters before -> "
          f"{len(after_sets)} clusters after. "
          f"{new_count} new, {gone_count} disappeared.")


def cmd_diff() -> None:
    if not os.path.exists(_SNAPSHOT_PATH):
        print("No previous snapshot found — run `feed` at least once first.")
        return
    if not os.path.exists(_STATE_PATH):
        print("No exploration state found — run `feed` at least once first.")
        return

    with open(_SNAPSHOT_PATH, encoding="utf-8") as f:
        before = json.load(f)
    learner = PositionalClusterLearner.load(_STATE_PATH)
    after = _snapshot(learner)

    # Backward-compat: older snapshot files (pre-particle-tracking) are
    # flat {cid: members} for action clusters only, with no "particle"
    # key at all.
    if "action" in before or "particle" in before:
        before_action = before.get("action", {})
        before_particle = before.get("particle", {})
    else:
        before_action = before
        before_particle = {}

    _print_cluster_diff(before_action, after["action"], "ACTION")
    print()
    _print_cluster_diff(before_particle, after["particle"], "PARTICLE")


def cmd_query(sentence: str) -> None:
    learner = _load_or_init()
    print(f"Sentence: {sentence!r}")
    print(f"  tag_sentence: {learner.tag_sentence(sentence)}")
    print(f"  spo:          {learner.spo(sentence)}")
    for i, clause in enumerate(learner.spo_all(sentence)):
        print(f"  spo_all[{i}]:    {clause}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "status":
        cmd_status()
    elif cmd == "feed":
        if len(sys.argv) < 3:
            print("Usage: explore_clusters.py feed <path-or-->")
            sys.exit(1)
        cmd_feed(sys.argv[2])
    elif cmd == "feed-many":
        if len(sys.argv) < 3:
            print("Usage: explore_clusters.py feed-many <file1> <file2> ...")
            sys.exit(1)
        cmd_feed_many(sys.argv[2:])
    elif cmd == "diff":
        cmd_diff()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: explore_clusters.py query \"<kalimat>\"")
            sys.exit(1)
        cmd_query(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
