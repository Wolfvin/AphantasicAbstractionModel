#!/usr/bin/env python3
"""Interactive REPL — "chat" directly with the exploration learner.

Loads the exploration state ONCE, then loops reading sentences from
stdin and prints, for EACH query:
  - tag_sentence (per-token grammar-class tags)
  - spo / spo_embedded (subject-predicate-object, with recursive
    embedded-clause detection — Round 25)
  - for the recognised predicate: which cluster it landed in + that
    cluster's top co-occurring objects (so you can see WHY AGNN put
    it there, not just THAT it did)
  - for every OTHER token: action/particle classification + cluster
    membership, so you can spot a pattern AGNN found that you didn't
    expect (or confirm one you did).

This is read-only — never calls train()/save(). Use explore_clusters.py
feed to grow the corpus; use this script to inspect what's already
learned.

Usage (run from AGNN/):
    python chat_agnn.py
    > kucing makan ikan
    ... (printed analysis) ...
    > exit
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explore_clusters import _load_or_init
from neocortex.positional_cluster_learner import PositionalClusterLearner


def _describe_token(learner: PositionalClusterLearner, token: str) -> str:
    """One-line summary of what AGNN currently believes about ``token``."""
    if learner._is_action_token(token):
        cid = learner.cluster_id_of.get(token)
        if cid is None or cid < 0:
            return f"{token!r}: ACTION (belum masuk cluster — di bawah min_action_observations)"
        members = sorted(learner.action_clusters.get(cid, []))
        siblings = [m for m in members if m != token][:6]
        objs = learner.action_object_freq.get(token, {})
        top_objs = sorted(objs.items(), key=lambda x: -x[1])[:5]
        return (
            f"{token!r}: ACTION, cluster {cid} (size={len(members)}) "
            f"bareng {siblings}{'...' if len(members) > 7 else ''} | "
            f"objek umum token ini: {top_objs}"
        )
    if learner._is_particle_token(token):
        pid = learner.particle_cluster_id_of.get(token)
        label = learner._particle_label_for(token)
        if pid is None or pid < 0:
            return f"{token!r}: PARTICLE (soft — belum masuk particle cluster formal)"
        members = sorted(learner.particle_clusters.get(pid, []))[:8]
        return f"{token!r}: PARTICLE, cluster {pid} ({label or 'belum dilabel'}), bareng {members}"
    sc = learner.object_supercluster_id.get(token)
    if sc is not None:
        siblings = sorted(learner.object_superclusters.get(sc, []))[:6]
        return f"{token!r}: OBJECT, super-cluster {sc}, bareng {siblings}"
    pf = learner.positional_freq.get(token)
    if pf:
        return f"{token!r}: UNKNOWN, positional_freq={pf} (belum cukup sinyal untuk kategori apapun)"
    return f"{token!r}: BELUM PERNAH DILIHAT (out-of-vocabulary)"


def chat(sentence: str, learner: PositionalClusterLearner) -> None:
    print(f"\n>>> {sentence!r}")
    print(f"  tag_sentence : {learner.tag_sentence(sentence)}")

    flat = learner.spo(sentence)
    print(f"  spo          : subject={flat.subject!r} predicate={flat.predicate!r} "
          f"object={flat.object!r} negated={flat.negated}")

    rich = learner.spo_embedded(sentence)
    if rich.embedded is not None:
        print(f"  spo_embedded : OBJEK ADALAH KLAUSA EMBEDDED -> "
              f"subject={rich.embedded.subject!r} predicate={rich.embedded.predicate!r} "
              f"object={rich.embedded.object!r}")
    else:
        print("  spo_embedded : objek flat (bukan klausa embedded)")

    print("  --- per-token ---")
    tokens = learner._tokenize(sentence)
    for tok in tokens:
        print(f"    {_describe_token(learner, tok)}")


def main() -> None:
    print("Memuat exploration state...")
    learner = _load_or_init()
    print(f"Siap. {len(learner.positional_freq)} token, "
          f"{len(learner.action_object_freq)} action, "
          f"{len(learner.action_clusters)} action cluster, "
          f"{len(learner.particle_clusters)} particle cluster.")
    print("Ketik kalimat (bahasa Indonesia), atau 'exit'/'quit' untuk keluar.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit", "q"}:
            break
        chat(line, learner)


if __name__ == "__main__":
    main()
