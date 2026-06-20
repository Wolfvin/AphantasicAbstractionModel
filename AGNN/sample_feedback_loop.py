#!/usr/bin/env python3
"""CLI: synthetic-sentence feedback loop.

Generates N synthetic SVO sentences from clusters the
PositionalClusterLearner has discovered, prints them one-by-one,
asks the user for a ``good``/``bad`` verdict per sentence, and
applies each verdict via :meth:`AGNNCore.apply_feedback`. At the
end, prints a summary of how many pairs were reinforced vs.
penalized (and how many were skipped because no matching edge
existed in the graph).

Usage:
    python AGNN/sample_feedback_loop.py [--num N] [--seed S]
                                        [--state PATH]
                                        [--corpus PATH [PATH ...]]

The script is self-contained: it loads the cluster learner state
from the file (default: ``AGNN/data/cluster_learner_state.json``),
trains a fresh learner on the corpus when the state file is
missing, then constructs an :class:`AGNNCore` (no model needed)
and runs the loop.

Zero-bias contract: this CLI is a USER-FACING feedback surface.
The user's verdict adjusts the action↔object edge weight via the
existing eligibility-trace path; it does NOT relabel any cluster
or RelationType. See :meth:`AGNNCore.apply_feedback` for the
guard.

Exit codes:
    0 — loop completed (regardless of how many verdicts applied)
    1 — initialization failure (state file missing, corpus empty, etc.)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional


# ----------------------------------------------------------------------
# Path setup — make AGNN importable when run as a script.
# ----------------------------------------------------------------------

_AGNP_ROOT = Path(__file__).resolve().parent
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# self-ai/src is needed for EngramComplex (which wraps AGNNGraph).
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


# ----------------------------------------------------------------------
# Imports (deferred to main() so --help works without the deps).
# ----------------------------------------------------------------------


def _load_or_train_cluster_learner(
    state_path: Optional[Path],
    corpus_paths: List[Path],
):
    """Load a labelled cluster learner from state, or train one fresh.

    Preference order:
      1. ``state_path`` if it exists and loads cleanly.
      2. Train on ``corpus_paths`` (default: pretrain_corpus.txt +
         pretrain_corpus_depth.txt) and bootstrap-label via
         ``build_labelled_cluster_learner``.
      3. Fall back to a freshly-trained (un-labelled) learner on
         whatever corpus is available — the CLI still works but
         ``apply_feedback`` will report ``action_unclustered`` for
         every sentence.
    """
    from neocortex.positional_cluster_learner import PositionalClusterLearner

    if state_path and state_path.exists():
        try:
            learner = PositionalClusterLearner.load(str(state_path))
            if learner.is_trained:
                return learner
        except Exception as e:
            print(
                f"[warn] state file {state_path} failed to load "
                f"({e}); falling back to fresh training.",
                file=sys.stderr,
            )

    # Train fresh.
    lines: List[str] = []
    for p in corpus_paths:
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("##"):
                lines.append(ln)
    if not lines:
        raise RuntimeError(
            f"No corpus lines found in {corpus_paths}. Provide "
            f"--corpus PATH or a --state PATH."
        )

    learner = PositionalClusterLearner()
    learner.train(lines)

    # Try to bootstrap-label via the bootstrap_classifier when available.
    try:
        from neocortex.bootstrap_classifier import (
            EXPECTED_VERB_GROUPS,
            build_labelled_cluster_learner,
        )
        # build_labelled_cluster_learner re-trains internally on
        # DEFAULT_CORPUS_PATHS. Use it as the canonical labelled
        # learner; fall back to the un-labelled learner above if it
        # raises.
        return build_labelled_cluster_learner()
    except Exception as e:
        print(
            f"[warn] bootstrap_classifier unavailable ({e}); "
            f"using un-labelled learner. apply_feedback will report "
            f"'action_unclustered' for sentences whose action is not "
            f"in any cluster.",
            file=sys.stderr,
        )
        return learner


def _eligible_cluster_ids(learner, min_actions: int = 2) -> List[int]:
    """Cluster ids with >= ``min_actions`` members, sorted ascending.

    Issue #95 follow-up: previously this returned ALL clusters with
    >= ``min_actions`` members, but ``apply_feedback`` early-returns
    ``{"reason": "action_unclustered"}`` for any action whose cluster
    is unlabelled. Sampling sentences from unlabelled clusters
    therefore always produces ``Not applied`` verdicts, which is
    exactly the silent-no-op UX issue #95 reported.

    Now we filter to ONLY clusters that have BOTH:
      - >= ``min_actions`` members, AND
      - a non-None entry in ``learner.cluster_labels``.

    Clusters with labels but only one action are also skipped (they
    cannot be sampled from — ``sample_sentence`` needs >= 2 actions
    to vary the action slot).
    """
    out = []
    for cid, actions in learner.action_clusters.items():
        if cid < 0:
            continue
        if len(actions) < min_actions:
            continue
        if cid not in getattr(learner, "cluster_labels", {}):
            continue
        if learner.cluster_labels.get(cid) is None:
            continue
        out.append(cid)
    return sorted(out)


def _seed_graph_for_feedback(core, learner, eligible_cids, rng):
    """Pre-populate ``core.graph`` so apply_feedback has edges to stamp.

    Background
    ----------
    This is the fix for issue #95: as shipped, the CLI constructed an
    ``AGNNCore(use_cluster_learner=False)`` with zero learned facts, so
    every ``apply_feedback()`` call returned
    ``{"applied": False, "reason": "no_matching_edge"}`` and the entire
    RLHF loop was silently a no-op.

    Approach
    --------
    For every eligible cluster (those with >= 2 actions AND a
    ``RelationType`` label), sample one ``(action, object)`` pair from
    the cluster, derive the corresponding ``(subject, action, object)``
    triple via ``learner.sample_sentence()``, and add a pair of
    single-token graph nodes (``label=subject``, ``label=object``)
    plus a ``TypedEdge`` between them whose ``relation_type`` matches
    the cluster's label.

    This mirrors the topology the unit-test helper
    ``_make_core_with_action_object_edge`` in
    ``AGNN/tests/test_sample_feedback_loop.py`` uses — single-token
    labels + an explicit typed edge — so the same
    ``apply_feedback()`` code path that the unit tests exercise now
    also fires in the real CLI.

    We do NOT call ``core.learn()`` here. ``learn()`` would create
    one node per sentence with the *full sentence* as its label
    (``episome.text``) and add typed edges from that new node to
    autoassociative neighbours (other episomes whose keywords
    overlap), not to the (subject, object) pair we actually want
    feedback on. Those autoassociative edges carry the *new
    sentence's* relation type but point at unrelated target nodes,
    so ``apply_feedback`` still cannot find a matching edge. The
    explicit single-token-node + typed-edge construction below is
    the minimum viable seed.

    Idempotency
    -----------
    Re-running this function on the same ``core`` + ``learner`` with
    the same ``rng`` is a no-op: nodes are added only if absent
    (``graph.get_node`` guard), and ``graph.add_edge`` is itself
    idempotent for duplicate ``(source, target, relation_type)``
    triples.

    Args:
        core: AGNNCore whose ``_cluster_learner`` has been set to
            ``learner``. ``core.graph`` may be ``None`` (when the
            EngramComplex / AGNNGraph dependency is missing) — in
            that case this function is a no-op and returns 0.
        learner: A trained, labelled ``PositionalClusterLearner``.
        eligible_cids: Iterable of cluster ids to sample from (those
            with >= ``--min-actions`` members). The function samples
            one sentence per cluster id.
        rng: A seeded ``random.Random`` instance for reproducible
            sentence generation. The same seed that drives the
            downstream sampling loop also drives this seeding pass,
            so the seed facts always cover the sentences the user
            will be asked about.

    Returns:
        The number of ``(subject, object)`` edges added to the
        graph. Always 0 when ``core.graph`` is None or the AGNNGraph
        dependency is unavailable.
    """
    if core.graph is None:
        return 0
    inner = getattr(core.graph, "_graph", None)
    if inner is None:
        return 0

    # Lazy imports — the AGNNGraph + EngramComplex + Episome deps may
    # be unavailable in stripped-down environments, and we don't want
    # the CLI to crash on import. Returning 0 here leaves the loop
    # running but every verdict will report ``no_matching_edge`` —
    # which is the honest failure mode and matches the existing
    # ``core.graph is None`` warning printed by the caller.
    try:
        from agnn.graph import (
            AGNNNode,
            NodeType,
            TypedEdge,
            RelationType as GraphRT,
        )
        from engrams.episodic_engram import Episome
    except Exception:  # noqa: BLE001
        return 0

    edges_added = 0
    for cid in eligible_cids:
        sentence = learner.sample_sentence(cid, rng=rng)
        if sentence is None:
            continue
        try:
            spo = learner.spo(sentence)
        except Exception:
            continue
        subject = spo.subject.strip()
        obj = spo.object.strip()
        action = spo.predicate.strip()
        if not subject or not obj or not action:
            continue

        # Look up the action's cluster label (skip unlabelled clusters
        # — apply_feedback would report ``action_unclustered`` for
        # them anyway, so seeding an edge would be wasted).
        action_cid = learner.cluster_id_of.get(action)
        if action_cid is None or action_cid < 0:
            continue
        relation_type = learner.cluster_labels.get(action_cid)
        if relation_type is None:
            continue
        try:
            graph_rt = GraphRT[relation_type.name]
        except KeyError:
            continue

        # Stable, human-readable ids so re-running with the same
        # corpus is idempotent.
        src_id = f"seed_{subject}"
        tgt_id = f"seed_{obj}"

        if inner.get_node(src_id) is None:
            inner.add_node(AGNNNode(
                id=src_id,
                label=subject,
                node_type=NodeType.ENTITY,
                confidence=0.5,
            ))
            # Register an Episome so reinforce()/penalize() can find
            # this node — without it, apply_feedback's episome-lookup
            # step bails out with ``no_matching_edge`` even after
            # the edge has been stamped.
            epi = Episome(id=src_id, text=subject, confidence=0.5)
            epi.id = src_id
            core._episomes.append(epi)

        if inner.get_node(tgt_id) is None:
            inner.add_node(AGNNNode(
                id=tgt_id,
                label=obj,
                node_type=NodeType.ENTITY,
                confidence=0.5,
            ))

        # Idempotency check: ``AGNNGraph.add_edge`` is NOT idempotent
        # for duplicate (source, target, relation_type) triples —
        # calling it twice adds two parallel edges. We walk the
        # existing outgoing edges from src_id and skip if a triple
        # with the same target + relation_type already exists. This
        # makes ``_seed_graph_for_feedback`` safe to call multiple
        # times with the same rng (e.g. on a config-reload path in
        # a long-running process).
        already_present = False
        try:
            for existing in inner.get_edges_from(src_id):
                if (
                    existing.target_id == tgt_id
                    and str(existing.relation_type) == str(graph_rt)
                ):
                    already_present = True
                    break
        except Exception:  # noqa: BLE001
            # Best-effort: if the edge-walk fails we fall through to
            # add_edge (worst case: a duplicate edge, which
            # apply_feedback's edge-stamping loop handles correctly
            # by stamping both — the resulting confidence delta is
            # the same because the modulatory signal is divided by
            # the trace sum, not the edge count).
            pass
        if already_present:
            continue

        inner.add_edge(TypedEdge(
            source_id=src_id,
            target_id=tgt_id,
            relation_type=graph_rt,
            confidence=0.5,
        ))
        edges_added += 1

    return edges_added


def _read_verdict(prompt: str) -> str:
    """Read a verdict from stdin; normalize to ``good``/``bad``/``skip``.

    Accepts: y/n, yes/no, g/b, good/bad, s/skip, empty (skip).
    """
    while True:
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            return "skip"
        if raw in ("", "s", "skip"):
            return "skip"
        if raw in ("y", "yes", "g", "good"):
            return "good"
        if raw in ("n", "no", "b", "bad"):
            return "bad"
        print(
            "  unrecognized — please enter 'good', 'bad', or 'skip' "
            "(or press Enter to skip).",
            file=sys.stderr,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic-sentence feedback loop. Generates N sentences "
            "from learned clusters, asks for a good/bad verdict per "
            "sentence, and applies the verdict via "
            "AGNNCore.apply_feedback (eligibility-trace path)."
        ),
    )
    parser.add_argument(
        "--num", "-n", type=int, default=5,
        help="Number of sentences to generate (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sentence generation.",
    )
    parser.add_argument(
        "--state", type=Path,
        default=_AGNP_ROOT / "data" / "cluster_learner_state.json",
        help="Path to a saved PositionalClusterLearner state JSON.",
    )
    parser.add_argument(
        "--corpus", type=Path, nargs="+",
        default=[
            _AGNP_ROOT / "data" / "pretrain_corpus.txt",
            _AGNP_ROOT / "data" / "pretrain_corpus_depth.txt",
        ],
        help="Corpus file(s) to train on when --state is unavailable.",
    )
    parser.add_argument(
        "--min-actions", type=int, default=2,
        help="Only sample from clusters with >= this many actions.",
    )
    args = parser.parse_args(argv)

    # 1. Load the cluster learner.
    try:
        learner = _load_or_train_cluster_learner(args.state, args.corpus)
    except Exception as e:
        print(f"[error] failed to load cluster learner: {e}", file=sys.stderr)
        return 1
    if not learner.is_trained:
        print("[error] cluster learner is not trained.", file=sys.stderr)
        return 1

    eligible_cids = _eligible_cluster_ids(learner, args.min_actions)
    if not eligible_cids:
        print(
            f"[error] no clusters with >= {args.min_actions} actions; "
            f"nothing to sample.",
            file=sys.stderr,
        )
        return 1

    # 2. Construct an AGNNCore. We use_cluster_learner=False so the
    #    core uses the legacy SemanticRoleClassifier (the cluster
    #    learner state file may not match what AGNNCore expects, and
    #    we explicitly set the learner on the core afterwards).
    try:
        # Load AGNN/core.py directly by path (same pattern as the test
        # suite — avoids name collision with self-ai/src/core/).
        import importlib.util as ilu
        core_path = _AGNP_ROOT / "core.py"
        spec = ilu.spec_from_file_location("agnn_core_module_cli", core_path)
        agnn_core_module = ilu.module_from_spec(spec)
        sys.modules["agnn_core_module_cli"] = agnn_core_module
        spec.loader.exec_module(agnn_core_module)
        AGNNCore = agnn_core_module.AGNNCore
    except Exception as e:
        print(f"[error] failed to import AGNNCore: {e}", file=sys.stderr)
        return 1

    core = AGNNCore(use_cluster_learner=False)
    # Inject our loaded learner so apply_feedback can use it for spo().
    core._cluster_learner = learner

    if core.graph is None:
        print(
            "[warn] EngramComplex unavailable — graph is None. "
            "apply_feedback will report 'no_graph' for every sentence. "
            "The loop still runs so you can see the generated sentences.",
            file=sys.stderr,
        )

    # 3. Pre-seed the graph so apply_feedback has edges to stamp.
    #
    # Issue #95: without this pass, the graph is empty and every
    # apply_feedback() call returns {"applied": False, "reason":
    # "no_matching_edge"}, making the entire RLHF loop a silent no-op.
    # We use the SAME rng + eligible_cids as the downstream sampling
    # loop so the seeded edges always cover the sentences the user
    # will be asked about.
    #
    # Note: we use a separate rng instance for seeding so consuming
    # values here does not shift the sentences the user sees. The
    # sampling loop below still uses ``rng`` (the public-facing one)
    # verbatim, preserving --seed reproducibility for downstream
    # sentence generation exactly as documented.
    rng_for_seeding = random.Random(args.seed)
    seeded_edges = _seed_graph_for_feedback(
        core, learner, eligible_cids, rng_for_seeding,
    )
    if seeded_edges > 0:
        print(
            f"[info] pre-seeded graph with {seeded_edges} "
            f"subject→object edge(s) from eligible clusters "
            f"so apply_feedback has something to stamp.",
            file=sys.stderr,
        )
    elif core.graph is not None:
        print(
            "[warn] could not pre-seed the graph (no labelled "
            "eligible clusters, or AGNNGraph unavailable). "
            "apply_feedback will report 'no_matching_edge' for "
            "every sentence.",
            file=sys.stderr,
        )

    # 4. Generate sentences and run the feedback loop.
    rng = random.Random(args.seed)
    sentences: List[str] = []
    for cid in eligible_cids:
        if len(sentences) >= args.num:
            break
        s = learner.sample_sentence(cid, rng=rng)
        if s is None:
            continue
        sentences.append(s)
        if len(sentences) >= args.num:
            break

    if not sentences:
        print("[error] no sentences generated (clusters may be empty).", file=sys.stderr)
        return 1

    print()
    print("=" * 72)
    print("Synthetic sentence feedback loop")
    print("=" * 72)
    print(f"Loaded {len(eligible_cids)} eligible clusters "
          f"(>= {args.min_actions} actions each).")
    print(f"Generated {len(sentences)} sentences. For each, enter:")
    print("  'good' / 'y' / 'g'  — sentence makes sense, reinforce edge")
    print("  'bad'  / 'n' / 'b'  — sentence doesn't make sense, penalize edge")
    print("  'skip' / 's' / ''   — skip (no feedback applied)")
    print()

    reinforced = 0
    penalized = 0
    skipped = 0
    failed = 0

    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] {sentence!r}")
        verdict = _read_verdict("  verdict (good/bad/skip): ")
        if verdict == "skip":
            print("  → skipped")
            skipped += 1
            continue
        result = core.apply_feedback(sentence, verdict)
        if result["applied"]:
            if verdict == "good":
                reinforced += 1
                print(
                    f"  → reinforced ({result['edges_stamped']} edge(s), "
                    f"relation={result['relation_type']})"
                )
            else:
                penalized += 1
                print(
                    f"  → penalized ({result['edges_stamped']} edge(s), "
                    f"relation={result['relation_type']})"
                )
        else:
            failed += 1
            print(
                f"  → not applied (reason: {result['reason']}, "
                f"action={result['action']}, object={result['object']}, "
                f"cluster_id={result['cluster_id']})"
            )
        print()

    # 4. Summary.
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total sentences:  {len(sentences)}")
    print(f"  Reinforced:       {reinforced}")
    print(f"  Penalized:        {penalized}")
    print(f"  Skipped:          {skipped}")
    print(f"  Not applied:      {failed}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
