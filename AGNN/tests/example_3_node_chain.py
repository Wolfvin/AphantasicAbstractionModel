"""
Concrete end-to-end example of BA 44 deductive reasoning.

Demonstrates a 3-node chain: Socrates -> human -> mortal (CATEGORICAL).
Run from repo root:
    python AGNN/tests/example_3_node_chain.py
"""

import os
import sys
from pathlib import Path

_AGNN_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNN_ROOT))

from engrams.episodic_engram import Episome
from engrams.semantic_engram import Semesome
from neocortex.inferior_frontal_gyrus import (
    CATEGORICAL,
    InferiorFrontalGyrus,
)


def main():
    # 3 Episome nodes (single-fact memory units).
    socrates = Episome(id=1, text="Socrates", confidence=1.0)
    human = Episome(id=2, text="human", confidence=1.0)
    mortal = Episome(id=3, text="mortal", confidence=1.0)

    # 2 Semesome edges forming a 3-node chain.
    edges = [
        Semesome(type=CATEGORICAL, weight=1.0,
                 source=socrates.text, target=human.text),
        Semesome(type=CATEGORICAL, weight=1.0,
                 source=human.text, target=mortal.text),
    ]

    # BA 44 deductive reasoning.
    ba44 = InferiorFrontalGyrus()
    deduction = ba44.deduce(edges)

    print("=" * 60)
    print("BA 44 DEDUCTIVE REASONING - 3-NODE CHAIN EXAMPLE")
    print("=" * 60)
    print(f"Input edges: {len(edges)}")
    for e in edges:
        print(f"  {e.source} -> {e.target} ({e.type} {e.weight})")
    print()
    print(f"Rule firings: {deduction.rule_count}")
    print(f"Applied rules: {deduction.applied_rules}")
    print(f"Aggregate confidence: {deduction.confidence:.4f}")
    print()
    print("Inferred edges:")
    for e in deduction.inferred_edges:
        print(f"  {e.source} -> {e.target} ({e.type} {e.weight})")
    print()
    print("Reasoning trace:")
    print(deduction.context)
    print("=" * 60)


if __name__ == "__main__":
    main()
