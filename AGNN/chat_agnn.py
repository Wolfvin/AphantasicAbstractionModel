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

Plain text -> sentence analysis (as above). Special commands:
  :trace <token>     Show the Q/K/V scoring process behind a token's
                      CURRENT cluster placement — its query vector and
                      cosine similarity against every existing
                      cluster's centroid, ranked, with the threshold
                      and the winning decision made visible.
  :try <sentence>     Trial-feed this sentence into an IN-MEMORY COPY
                      of the learner (full single-call rebuild over
                      the real history + this sentence — same fix as
                      Round 36) and print which clusters changed,
                      WITHOUT saving anything to disk. Repeatable —
                      each :try adds to a staged batch.
  :diff               Re-print the last :try's before/after diff.
  :commit <label>     Persist the staged :try batch for real via
                      explore_clusters.py's normal feed pipeline
                      (writes state + appends to the feed log).
  :reset              Discard the staged :try batch without committing.

This script is read-only with respect to the SAVED state — :try never
writes to disk; only :commit does, and only when you explicitly ask.

Usage (run from AGNN/):
    python chat_agnn.py
    > kucing makan ikan
    > :trace makan
    > :try kucing mengeong setiap pagi sebelum makan.
    > :commit testing kalimat baru
    > exit
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import explore_clusters as ec
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


def _supercluster_vector(learner: PositionalClusterLearner, token: str) -> Dict[int, int]:
    """Project token's literal object-count map to super-cluster ids —
    the SAME feature vector ``_cluster_actions`` builds internally,
    replicated here read-only for display (no state mutated).
    """
    objs = learner.action_object_freq.get(token, {})
    sc_map: Dict[int, int] = {}
    for obj, count in objs.items():
        sc_id = learner.object_supercluster_id.get(obj)
        if sc_id is None:
            sc_id = hash(obj)
        sc_map[sc_id] = sc_map.get(sc_id, 0) + count
    return sc_map


def trace_token(learner: PositionalClusterLearner, token: str) -> None:
    """Show the Q/K/V scoring process behind ``token``'s CURRENT
    cluster placement: its query vector vs every existing cluster's
    centroid, ranked by cosine similarity, with the threshold and the
    actual winning decision made visible. Pure read/replay — does not
    re-run clustering, just recomputes the same score formula
    ``_cluster_action_group_qkv`` uses internally.
    """
    if token not in learner.action_object_freq:
        print(f"  '{token}' tidak ada di action_object_freq — belum pernah "
              f"diekstrak sebagai action sama sekali, tidak bisa di-trace.")
        return

    query = _supercluster_vector(learner, token)
    print(f"  Query vector (super-cluster) untuk {token!r}: {query}")
    print(f"  Threshold qkv_action_similarity_threshold: "
          f"{learner.qkv_action_similarity_threshold}")

    my_cid = learner.cluster_id_of.get(token)
    scores: List[Tuple[int, float, List[str]]] = []
    for cid, members in learner.action_clusters.items():
        other_members = [m for m in members if m != token]
        if not other_members:
            continue
        centroid: Dict[int, int] = {}
        for m in other_members:
            for sc_id, count in _supercluster_vector(learner, m).items():
                centroid[sc_id] = centroid.get(sc_id, 0) + count
        centroid_mean = {k: v / len(other_members) for k, v in centroid.items()}
        score = learner._cosine_similarity_sparse(query, centroid_mean)
        scores.append((cid, score, sorted(other_members)[:5]))

    scores.sort(key=lambda x: -x[1])
    print(f"  Top-5 cluster (ranking cosine similarity terhadap centroid):")
    for cid, score, sample in scores[:5]:
        crosses = "YA" if score >= learner.qkv_action_similarity_threshold else "tidak"
        you_are_here = " <-- {} BERADA DI SINI SEKARANG".format(repr(token)) if cid == my_cid else ""
        print(f"    cluster {cid:4d}  score={score:.4f}  lolos_threshold={crosses}  "
              f"contoh_anggota={sample}{you_are_here}")
    if my_cid is None or my_cid < 0:
        print(f"  Status sekarang: {token!r} BELUM masuk cluster manapun "
              f"(di bawah min_action_observations atau singleton-belum-cocok).")


_staged_lines: List[str] = []
_last_trial: Optional[PositionalClusterLearner] = None


def try_feed(sentence: str, base_learner: PositionalClusterLearner) -> None:
    """Trial-feed ``sentence`` into an IN-MEMORY rebuild — full history
    + every staged :try line so far + this new one — and print a
    before/after diff. Writes NOTHING to disk (no train()/save() on
    the real state). Uses the SAME single-call-rebuild fix as Round 36
    (_rebuild_from_scratch) so the trial result matches what a real
    commit would produce.
    """
    global _last_trial
    _staged_lines.append(sentence)
    print(f"  [TRIAL — belum disimpan] menambah {len(_staged_lines)} baris "
          f"staged, rebuild penuh atas riwayat + staged batch...")
    trial = ec._rebuild_from_scratch(_staged_lines)
    _last_trial = trial
    show_diff(base_learner, trial)


def show_diff(before: PositionalClusterLearner, after: PositionalClusterLearner) -> None:
    before_sets = {frozenset(v): cid for cid, v in before.action_clusters.items()}
    after_sets = {frozenset(v): cid for cid, v in after.action_clusters.items()}
    new_clusters = set(after_sets) - set(before_sets)
    gone_clusters = set(before_sets) - set(after_sets)

    print(f"  --- Diff cluster ACTION (sebelum vs sesudah trial) ---")
    print(f"  total cluster: {len(before_sets)} -> {len(after_sets)}")
    if new_clusters:
        print(f"  CLUSTER BARU/BERUBAH ({len(new_clusters)}):")
        for c in sorted(new_clusters, key=lambda s: -len(s))[:10]:
            print(f"    + {sorted(c)}")
    if gone_clusters:
        print(f"  CLUSTER HILANG/BERUBAH ({len(gone_clusters)}):")
        for c in sorted(gone_clusters, key=lambda s: -len(s))[:10]:
            print(f"    - {sorted(c)}")
    if not new_clusters and not gone_clusters:
        print("  Tidak ada perubahan komposisi cluster action sama sekali.")

    before_particles = {frozenset(v) for v in before.particle_clusters.values()}
    after_particles = {frozenset(v) for v in after.particle_clusters.values()}
    if before_particles != after_particles:
        print(f"  Particle cluster JUGA berubah: {len(before_particles)} -> "
              f"{len(after_particles)}")


def main() -> None:
    print("Memuat exploration state...")
    learner = _load_or_init()
    print(f"Siap. {len(learner.positional_freq)} token, "
          f"{len(learner.action_object_freq)} action, "
          f"{len(learner.action_clusters)} action cluster, "
          f"{len(learner.particle_clusters)} particle cluster.")
    print("Ketik kalimat (bahasa Indonesia), atau 'exit'/'quit' untuk keluar.\n")

    global _last_trial
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

        if line.startswith(":trace "):
            trace_token(learner, line[len(":trace "):].strip())
        elif line.startswith(":try "):
            try_feed(line[len(":try "):].strip(), learner)
        elif line == ":diff":
            if _last_trial is None:
                print("  Belum ada :try yang dijalankan.")
            else:
                show_diff(learner, _last_trial)
        elif line.startswith(":commit"):
            if not _staged_lines:
                print("  Tidak ada staged batch untuk di-commit.")
            else:
                label = line[len(":commit"):].strip() or "chat_agnn manual commit"
                ec._commit_batch(list(_staged_lines), batch_label=label)
                print(f"  Ter-commit {len(_staged_lines)} baris. Reload state...")
                learner = _load_or_init()
                _staged_lines.clear()
                _last_trial = None
        elif line == ":reset":
            _staged_lines.clear()
            _last_trial = None
            print("  Staged batch dibuang.")
        else:
            chat(line, learner)


if __name__ == "__main__":
    main()
