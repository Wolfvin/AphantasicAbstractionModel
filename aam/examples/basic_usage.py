#!/usr/bin/env python3
"""Basic RSVS usage example.

Demonstrates the core workflow: ingest text, query concepts,
compare similarity, and inspect the knowledge graph.

Usage:
    python basic_usage.py
"""

from rsvs import Rsvs


def main() -> None:
    # --- 1. Create an RSVS instance ---
    r = Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)
    print("Created RSVS instance with default config.\n")

    # --- 2. Ingest some text ---
    sentences = [
        "Batu adalah material keras yang ditemukan di alam.",
        "Granit adalah jenis batu beku yang sangat keras.",
        "Marmer adalah batu metamorf yang digunakan untuk patung.",
        "Kayu adalah material organik yang berasal dari pohon.",
        "Besi adalah logam yang keras dan kuat.",
        "Air adalah zat cair yang esensial untuk kehidupan.",
        "Sungai mengalir dari gunung ke laut membawa batu dan pasir.",
        "Tulang manusia tersusun dari kalsium dan kolagen.",
    ]

    for sentence in sentences:
        stats = r.ingest(sentence)
        print(
            f"  Ingested: {sentence[:50]}... "
            f"(atoms={stats.atoms_promoted}, senses={stats.sense_created})"
        )

    print(f"\nGraph status: {r.status()}\n")

    # --- 3. Query a concept ---
    result = r.query("batu", "material keras")
    if result:
        print(f"Query 'batu' in context 'material keras':")
        print(f"  Sense: {result.sense_idx}/{result.sense_n}")
        print(f"  Layer: {result.layer}")
        print(f"  Grounding: {result.grounding_score:.3f}")
        print(f"  Top atoms: {result.top_atoms(5)}")
        print(f"  Compositions: {result.compositions}")
    print()

    # --- 4. Similarity comparison ---
    sim = r.similarity("batu", "kayu")
    if sim:
        print(f"Similarity 'batu' vs 'kayu':")
        print(f"  Jaccard: {sim.jaccard:.3f}")
        print(f"  Shared: {sim.shared}")
        print(f"  Only 'batu': {sim.only_a}")
        print(f"  Only 'kayu': {sim.only_b}")
    print()

    # --- 5. Appraise text against the graph ---
    verdict = r.appraise("Batu adalah material yang sangat keras dan kuat")
    print(f"Appraise verdict: {verdict.verdict}")
    print(f"  Agree: {verdict.agree_pct:.1f}%  Disagree: {verdict.disagree_pct:.1f}%")
    print()

    # --- 6. Find related concepts ---
    related = r.relate("batu")
    if related:
        print(f"Concepts related to 'batu':")
        for node_id, score in related.related_nodes[:5]:
            print(f"  Node #{node_id}: score={score:.3f}")
    print()

    # --- 7. Inspect node details ---
    info = r.node_info("batu")
    print(f"Node info for 'batu':")
    print(f"  ID: {info.id}")
    print(f"  Confidence: {info.confidence:.3f}")
    print(f"  Tier: {info.tier}")
    print(f"  Status: {info.status}")
    print(f"  Layer: {info.layer}")
    print(f"  Is seed: {info.is_seed}")

    # --- 8. List all nodes ---
    all_nodes = r.nodes(include_seeds=False)
    print(f"\nAll non-seed nodes ({len(all_nodes)}): {all_nodes[:10]}...")

    # --- 9. Save and load ---
    r.save("/tmp/rsvs_basic_demo.json")
    print("\nSaved graph to /tmp/rsvs_basic_demo.json")

    r2 = Rsvs.load("/tmp/rsvs_basic_demo.json")
    status2 = r2.status()
    print(f"Loaded graph status: {status2}")


if __name__ == "__main__":
    main()
