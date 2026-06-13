#!/usr/bin/env python3
"""Structural reasoning with MCTS and context-aware queries.

This example demonstrates RSVS's reasoning capabilities:
- Monte Carlo Tree Search (MCTS) for exploring reasoning paths
- Context-aware queries with depth-controlled traversal
- Consolidation and reflection for graph maintenance
- Thinking mode for adjusting query depth

Usage:
    python structural_reasoning.py
"""

from rsvs import Rsvs


def main() -> None:
    r = Rsvs(entity_promote_n=2, theta_assign=0.10, n_warm=15, eta=0.1)

    # --- 1. Build a rich knowledge base ---
    corpus = [
        # Geology
        "Batu adalah material padat dari mineral.",
        "Granit adalah batu beku yang terbentuk dari magma.",
        "Marmer terbentuk dari batu gamping yang mengalami metamorfisme.",
        "Pasir adalah butiran kecil dari batu yang tererosi.",
        # Biology
        "Tulang adalah jaringan keras yang menyusun rangka manusia.",
        "Kalsium adalah mineral utama dalam tulang.",
        "Kolagen adalah protein yang memberi fleksibilitas pada tulang.",
        "Otak adalah organ pusat sistem saraf manusia.",
        # Physics
        "Kekerasan adalah ketahanan material terhadap deformasi.",
        "Mohs scale mengukur kekerasan mineral dari 1 sampai 10.",
        "Intan memiliki kekerasan 10 pada skala Mohs.",
        "Elastisitas adalah kemampuan material kembali ke bentuk semula.",
        # Materials
        "Besi adalah logam yang keras dan magnetis.",
        "Baja adalah paduan besi dengan karbon.",
        "Kaca adalah material transparan dari silika.",
        "Plastik adalah polimer sintetis yang ringan.",
    ]

    for sentence in corpus:
        r.ingest(sentence)

    print(f"Knowledge base built: {r.status()}\n")

    # --- 2. Standard query vs context-aware query ---
    # Standard query: simple context
    q_simple = r.query("batu", "material")
    if q_simple:
        print(f"Standard query 'batu' (context='material'):")
        print(f"  Sense: {q_simple.sense_idx}/{q_simple.sense_n}, Layer: {q_simple.layer}")
        print(f"  Top atoms: {q_simple.top_atoms(5)}")

    # Context-aware query: multiple context atoms with depth control
    cqr = r.context_query(
        "batu",
        context_atoms=["kekerasan", "mineral", "material"],
        max_depth=5,
        gamma=0.85,
        halt_confidence=0.7,
    )
    if cqr:
        print(f"\nContext-aware query 'batu' (context=['kekerasan', 'mineral', 'material']):")
        print(f"  Sense: {cqr.active_sense_idx}/{cqr.total_senses}, Layer: {cqr.layer}")
        print(f"  Depth reached: {cqr.depth_reached}")
        print(f"  Halt reason: {cqr.halt_reason}")
        print(f"  Cycles detected: {cqr.cycles_detected}")
        top_atoms = [f"{label}:{score:.3f}" for label, score in cqr.scored_atoms[:5]]
        print(f"  Top atoms: {top_atoms}")
    print()

    # --- 3. MCTS reasoning ---
    # Explore reasoning paths for "batu"
    mcts = r.mcts_query("batu", simulations=50, exploration=1.414)
    if mcts:
        print(f"MCTS query for 'batu':")
        print(f"  Active sense: {mcts.active_sense_idx}/{mcts.total_senses}")
        print(f"  Simulations run: {mcts.simulations_run}")
        print(f"  Depth reached: {mcts.depth_reached}")
        print(f"  Halt reason: {mcts.halt_reason}")
        print(f"  Best path: {mcts.best_path}")
        top_mcts = [f"{label}:{score:.3f}" for label, score in mcts.scored_atoms[:5]]
        print(f"  Top atoms: {top_mcts}")
    print()

    # --- 4. Structural comparison across domains ---
    # "batu" (geology) vs "tulang" (biology) — both are hard materials
    ssim = r.structural_similarity("batu", "tulang")
    if ssim:
        print(f"Structural similarity 'batu' vs 'tulang': {ssim.structural_similarity:.3f}")
        print(f"  Shared compositions: {len(ssim.shared_compositions)}")

    # Substitution: what transforms "batu" into "tulang"?
    sub = r.substitution_analysis("batu", "tulang")
    if sub:
        print(f"Substitution 'batu' -> 'tulang': {len(sub.substitutions)} substitution(s)")
        if sub.substitution_labels(r):
            print(f"  Substitutions: {sub.substitution_labels(r)[:3]}")
    print()

    # --- 5. Graph maintenance ---
    # Consolidate: merge similar senses, prune weak edges
    consolidation = r.consolidate()
    print(f"Consolidation:")
    print(f"  Senses merged: {consolidation.senses_merged}")
    print(f"  Senses removed: {consolidation.senses_removed}")
    print(f"  Edges pruned: {consolidation.edges_pruned}")
    print(f"  Atoms compacted: {consolidation.atoms_compacted}")

    # Reflect: periodic self-correction
    reflection = r.run_reflection()
    print(f"\nReflection:")
    print(f"  Actions total: {reflection.actions_total}")
    print(f"  Actions applied: {reflection.actions_applied}")

    # Verify graph integrity
    verify = r.verify()
    print(f"\nVerification: {verify}")

    # --- 6. Entity candidates ---
    candidates = r.entity_candidates(top_k=5)
    if candidates:
        print(f"\nEntity candidates (tokens that should be promoted to nodes):")
        for label, score in candidates[:5]:
            print(f"  '{label}': score={score:.3f}")

    # --- 7. Pending removals ---
    pending = r.pending_removals()
    if pending:
        print(f"\nPending removals (nodes needing approval): {pending}")
    else:
        print(f"\nNo pending removals.")


if __name__ == "__main__":
    main()
