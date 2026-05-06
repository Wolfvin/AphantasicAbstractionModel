#!/usr/bin/env python3
"""Composition demo — creating and analyzing compositional concepts.

RSVS's core innovation is compositional sense definitions: a concept
is defined by its components (atoms or other composed concepts),
not just by co-occurrence statistics.

This example demonstrates:
- Creating explicit compositions
- Analyzing structural similarity between composed concepts
- Substitution analysis (what transforms A into B?)
- Grounding evidence for composed senses

Usage:
    python composition_demo.py
"""

from rsvs import Rsvs


def main() -> None:
    r = Rsvs(entity_promote_n=2, theta_assign=0.10)

    # --- 1. Build a vocabulary of base concepts ---
    base_sentences = [
        "Laki-laki adalah manusia berjenis kelamin pria.",
        "Perempuan adalah manusia berjenis kelamin wanita.",
        "Kekuasaan adalah kemampuan untuk mempengaruhi orang lain.",
        "Negara adalah wilayah dengan pemerintahan tersendiri.",
        "Keraton adalah istana tempat raja tinggal.",
        "Hukum adalah aturan yang mengatur kehidupan bermasyarakat.",
        "Keluarga adalah unit terkecil dalam masyarakat.",
        "Rakyat adalah orang-orang yang tinggal di suatu negara.",
        "Perang adalah konflik bersenjata antar kelompok.",
        "Perdamaian adalah keadaan tanpa konflik.",
    ]

    for s in base_sentences:
        r.ingest(s)

    print("Base vocabulary ingested.\n")

    # --- 2. Create explicit compositions ---
    # "raja" = laki-laki + kekuasaan
    raja_id = r.compose("raja", [("laki-laki", 0), ("kekuasaan", 0)], lang="id")
    print(f"Composed 'raja' (id={raja_id}) = laki-laki + kekuasaan")

    # "ratu" = perempuan + kekuasaan
    ratu_id = r.compose("ratu", [("perempuan", 0), ("kekuasaan", 0)], lang="id")
    print(f"Composed 'ratu' (id={ratu_id}) = perempuan + kekuasaan")

    # "negarawan" = kekuasaan + negara + hukum
    negarawan_id = r.compose("negarawan", [("kekuasaan", 0), ("negara", 0)], lang="id")
    print(f"Composed 'negarawan' (id={negarawan_id}) = kekuasaan + negara")

    print()

    # --- 3. Structural similarity ---
    # raja and ratu share "kekuasaan" but differ in gender component
    ssim = r.structural_similarity("raja", "ratu")
    if ssim:
        print(f"Structural Similarity: 'raja' vs 'ratu'")
        print(f"  Score: {ssim.structural_similarity:.3f}")
        print(f"  Shared compositions: {len(ssim.shared_compositions)}")
        print(f"  Only 'raja': {len(ssim.only_a_compositions)}")
        print(f"  Only 'ratu': {len(ssim.only_b_compositions)}")
        print(f"  Layers: {ssim.layer_a}/{ssim.layer_b}")
        # Get human-readable labels for shared compositions
        if ssim.shared_labels(r):
            labels = ssim.shared_labels(r)
            print(f"  Shared labels: {labels}")
    print()

    # --- 4. Substitution analysis ---
    # What transforms "raja" into "ratu"?
    sub = r.substitution_analysis("raja", "ratu")
    if sub:
        print(f"Substitution Analysis: 'raja' -> 'ratu'")
        print(f"  Structural similarity: {sub.structural_similarity:.3f}")
        print(f"  Substitutions needed: {len(sub.substitutions)}")
        if sub.substitution_labels(r):
            labels = sub.substitution_labels(r)
            print(f"  Substitutions: {labels}")
        print(f"  Unpaired only 'raja': {len(sub.unpaired_only_a)}")
        print(f"  Unpaired only 'ratu': {len(sub.unpaired_only_b)}")
    print()

    # --- 5. Inspect composition details ---
    senses = r.senses("raja")
    if senses:
        s = senses[0]
        print(f"Senses for 'raja':")
        print(f"  Sense {s.sense_idx}: coherence={s.coherence:.3f}, layer={s.layer}")
        print(f"  Core atoms: {s.core_atoms}")
        print(f"  Compositions: {s.compositions}")
        print(f"  Grounding score: {s.grounding_score:.3f}")
        print(f"  Grounding evidence: confirming={s.grounding_evidence.confirming_contexts}, "
              f"contradicting={s.grounding_evidence.contradicting_contexts}")
    print()

    # --- 6. Context-weighted similarity ---
    # Compare "raja" and "negarawan" in the context of "kekuasaan"
    csim = r.context_similarity("raja", "negarawan", ["kekuasaan"])
    if csim is not None:
        print(f"Context similarity 'raja' vs 'negarawan' in context ['kekuasaan']: {csim:.3f}")

    # Compare same concepts in a different context
    csim2 = r.context_similarity("raja", "negarawan", ["negara", "hukum"])
    if csim2 is not None:
        print(f"Context similarity 'raja' vs 'negarawan' in context ['negara', 'hukum']: {csim2:.3f}")

    # --- 7. Verify the graph ---
    verify_result = r.verify()
    print(f"\nGraph verification: {verify_result}")


if __name__ == "__main__":
    main()
