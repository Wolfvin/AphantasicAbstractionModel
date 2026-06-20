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
    """Cluster ids with >= ``min_actions`` members, sorted ascending."""
    out = []
    for cid, actions in learner.action_clusters.items():
        if cid < 0:
            continue
        if len(actions) >= min_actions:
            out.append(cid)
    return sorted(out)


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

    # 3. Generate sentences and run the feedback loop.
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
