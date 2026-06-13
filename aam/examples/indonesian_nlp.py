#!/usr/bin/env python3
"""Indonesian NLP demo — using RSVS for Bahasa Indonesia semantic analysis.

RSVS prioritizes Bahasa Indonesia support. This example demonstrates:
- Ingesting Indonesian text
- Building a compositional knowledge graph
- Cross-domain semantic analysis
- Using the embedded corpus for quick start
- Domain-specific attention tuning

Usage:
    python indonesian_nlp.py
"""

from rsvs import Rsvs


def main() -> None:
    r = Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)

    # --- 1. Use the embedded Indonesian corpus ---
    # RSVS ships with a built-in corpus covering 7 domains
    try:
        from rsvs import DOMAINS, domain_names, get_domain_text

        print("Available domains:", domain_names())
        print(f"Total sentences in embedded corpus: {sum(len(v) for v in DOMAINS.values())}")

        # Ingest the geology domain
        geology_text = get_domain_text("geology")
        stats = r.ingest(geology_text)
        print(f"\nIngested geology domain: {stats.sentences_processed} sentences, "
              f"{stats.atoms_promoted} atoms promoted, {stats.sense_created} senses created")

        # Ingest the water domain
        water_text = get_domain_text("water")
        stats2 = r.ingest(water_text)
        print(f"Ingested water domain: {stats2.sentences_processed} sentences, "
              f"{stats2.atoms_promoted} atoms promoted")
    except ImportError:
        print("Embedded corpus not available, using manual sentences instead.")
        manual_sentences = [
            "Air mengalir dari pegunungan ke laut melalui sungai.",
            "Danau adalah kumpulan air di cekungan daratan.",
            "Hujan adalah presipitasi air dari atmosfer.",
            "Batu merupakan material padat penyusun kerak bumi.",
            "Mineral adalah zat padat anorganik dengan komposisi kimia tetap.",
        ]
        for s in manual_sentences:
            r.ingest(s)

    print()

    # --- 2. Set domain attention ---
    # Tune attention weights for each domain
    # Domain 0 = geology, Domain 1 = water (based on ingest order)
    r.set_domain_attention(domain_id=0, alpha=0.5, beta=0.3, gamma=0.2)
    r.set_domain_attention(domain_id=1, alpha=0.3, beta=0.4, gamma=0.3)
    print("Set domain attention weights for geology and water domains.\n")

    # --- 3. Query with Indonesian context ---
    result = r.query("air", "cair dan mengalir")
    if result:
        print(f"Query 'air' (context='cair dan mengalir'):")
        print(f"  Sense: {result.sense_idx}/{result.sense_n}, Layer: {result.layer}")
        print(f"  Top atoms: {result.top_atoms(5)}")
        print(f"  Grounding: {result.grounding_score:.3f}")
    print()

    # --- 4. Cross-domain comparison ---
    # Compare "air" (water domain) vs "batu" (geology domain)
    sim = r.similarity("air", "batu")
    if sim:
        print(f"Similarity 'air' vs 'batu': Jaccard={sim.jaccard:.3f}")
        print(f"  Shared: {sim.shared}")
        print(f"  Only 'air': {sim.only_a[:5]}")
        print(f"  Only 'batu': {sim.only_b[:5]}")

    # Structural similarity for composed concepts
    ssim = r.structural_similarity("air", "batu")
    if ssim:
        print(f"  Structural similarity: {ssim.structural_similarity:.3f}")
    print()

    # --- 5. Appraise Indonesian statements ---
    statements = [
        "Air adalah zat cair yang sangat penting untuk kehidupan.",
        "Batu bisa mengalir seperti air.",  # False statement
        "Mineral adalah komponen penyusun batu.",
    ]

    for stmt in statements:
        verdict = r.appraise(stmt)
        print(f"Appraise: \"{stmt[:60]}...\"")
        print(f"  Verdict: {verdict.verdict} "
              f"(agree={verdict.agree_pct:.1f}%, disagree={verdict.disagree_pct:.1f}%)")
    print()

    # --- 6. Compose Indonesian concepts ---
    # "sungai" = air + mengalir (river = water + flowing)
    try:
        sungai_id = r.compose("sungai", [("air", 0), ("mengalir", 0)], lang="id")
        print(f"Composed 'sungai' (id={sungai_id}) = air + mengalir")

        # Inspect the composition
        info = r.node_info("sungai")
        print(f"  Layer: {info.layer}, Confidence: {info.confidence:.3f}")
        print(f"  Derived from: {info.derived_from_node_ids}")
    except Exception as e:
        print(f"Composition not possible with current vocabulary: {e}")

    print()

    # --- 7. Context query for disambiguation ---
    # "batu" can mean rock (geology) or stone (material)
    cqr = r.context_query(
        "batu",
        context_atoms=["mineral", "geologi"],
        max_depth=4,
    )
    if cqr:
        print(f"Context query 'batu' (geology context):")
        print(f"  Depth: {cqr.depth_reached}, Halt: {cqr.halt_reason}")
        top = [f"{label}:{score:.3f}" for label, score in cqr.scored_atoms[:5]]
        print(f"  Top atoms: {top}")

    # --- 8. Save the Indonesian knowledge graph ---
    r.save("/tmp/rsvs_indonesian_demo.json")
    print("\nSaved Indonesian knowledge graph to /tmp/rsvs_indonesian_demo.json")

    # Final status
    status = r.status()
    print(f"Final graph: {int(status.get('total_nodes', 0))} nodes, "
          f"{int(status.get('total_atoms', 0))} atoms")


if __name__ == "__main__":
    main()
