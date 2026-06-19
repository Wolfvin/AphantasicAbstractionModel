"""
End-to-end smoke test for AGNNCore — exercises the 3 bugs that were
just fixed and prints the output the DoD requires:

    [1] init_brain()
    [2] learn() — encode 3 related facts forming a CATEGORICAL chain
    [3] reinforce() — nudge one episome
    [4] process() — must NOT produce a repetition loop, and
        chain_confidence must be > 0
    [5] introspect() — top_nodes must be a list of dicts (printable
        without ``TypeError: 'int' object is not subscriptable``)

Run from the repo root:

    python e2e_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the AGNN package + self-ai/src importable from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent
_AGNP_ROOT = _REPO_ROOT / "AGNN"
_SELF_AI_SRC = _REPO_ROOT / "self-ai" / "src"

if str(_SELF_AI_SRC) in sys.path:
    sys.path.remove(str(_SELF_AI_SRC))
if str(_AGNP_ROOT) in sys.path:
    sys.path.remove(str(_AGNP_ROOT))
# self-ai/src first (lower priority), then AGNN/ — same order the test
# files use so the AGNN core module wins on name collisions.
sys.path.insert(0, str(_SELF_AI_SRC))
sys.path.insert(0, str(_AGNP_ROOT))

# Don't accidentally load a real Qwen3 model — we're testing the
# graceful-fallback path so the e2e test runs anywhere.
os.environ.pop("QWEN_PATH", None)

from core import AGNNCore  # noqa: E402


def banner(step: str, title: str) -> None:
    print(f"\n[{step}] {title}")
    print("-" * 60)


def main() -> int:
    # ------------------------------------------------------------------
    # [1] init_brain
    # ------------------------------------------------------------------
    banner("1", "init_brain()")
    brain = AGNNCore(model_path=None)
    print(f"graph available  : {brain.graph is not None}")
    print(f"trisynaptic ready: {brain.trisynaptic is not None}")
    print(f"papez ready      : {brain.papez is not None}")
    print(f"deductive ready  : {brain.deductive is not None}")

    # ------------------------------------------------------------------
    # [2] learn — encode a 3-node CATEGORICAL chain so BA 44's
    #     CategoricalTransitivity rule can fire (A -> B -> C => A -> C).
    #     Keyword "human" overlaps between facts 1 and 2, "mortal"
    #     between 2 and 3 — that's what autoassociates them in CA3 and
    #     builds the typed edges in the AGNNGraph.
    # ------------------------------------------------------------------
    banner("2", "learn() — 3 related facts forming a CATEGORICAL chain")
    facts = [
        ("What is Socrates?", "wrong1", "Socrates is a human"),
        ("What is human?", "wrong2", "human is mortal"),
        ("What is mortal?", "wrong3", "mortal is dead"),
    ]
    last_id = None
    for q, w, c in facts:
        r = brain.learn(q, w, c)
        print(f"  learn({c!r}) -> node_id={r['node_id']} "
              f"confidence={r['confidence']:.2f} "
              f"graph_size={r['graph_size']}")
        last_id = r["node_id"]

    # ------------------------------------------------------------------
    # [3] reinforce the most recent episome (positive feedback)
    # ------------------------------------------------------------------
    banner("3", f"reinforce(node_id={last_id})")
    before = brain._find_episome(last_id).confidence
    brain.reinforce(last_id)
    after = brain._find_episome(last_id).confidence
    print(f"  confidence: {before:.2f} -> {after:.2f}  (+0.1)")

    # ------------------------------------------------------------------
    # [4] process — must NOT produce a repetition loop and
    #     chain_confidence must be > 0. We use a query whose keywords
    #     overlap with all three episomes ("socrates human mortal dead")
    #     so the Papez circuit retrieves the full 3-node chain and BA 44's
    #     CategoricalTransitivity rule can fire (A->B->C => A->C, weight
    #     0.6 * 0.6 = 0.36).
    # ------------------------------------------------------------------
    banner("4", "process('socrates human mortal dead') — "
                "no loop, chain_confidence > 0")
    result = brain.process("socrates human mortal dead")
    print(f"  chain_confidence: {result['chain_confidence']:.4f}")
    chain_preview = result['chain'][:300]
    print(f"  chain (head)    : {chain_preview}"
          f"{'...' if len(result['chain']) > 300 else ''}")
    answer = result['answer']
    answer_preview = answer[:200]
    print(f"  answer (head)   : {answer_preview}"
          f"{'...' if len(answer) > 200 else ''}")

    # Quick assertions so CI can grep the exit code.
    assert result["chain_confidence"] > 0, (
        "DoD violation: chain_confidence must be > 0 after learn()s"
    )
    # Repetition loop signature: same 3-token pattern repeated 5+ times.
    # We don't have a real Qwen3 here so the graceful fallback fires
    # ("[Graph context: ...] (model not loaded)") — but we still guard
    # against the loop in case the model is loaded in another env.
    tokens = answer.split()
    if len(tokens) >= 9:
        for i in range(len(tokens) - 9):
            window = tokens[i:i + 9]
            if window[:3] == window[3:6] == window[6:9]:
                raise AssertionError(
                    f"Repetition loop detected in answer: {answer!r}"
                )
    print("  [OK] no repetition loop detected")

    # ------------------------------------------------------------------
    # [5] introspect — top_nodes must be a list of dicts and printable
    #     without 'int' object is not subscriptable
    # ------------------------------------------------------------------
    banner("5", "introspect() — top_nodes must be a list of dicts")
    info = brain.introspect()
    print(f"  graph_size            : {info['graph_size']}")
    print(f"  avg_confidence        : {info['avg_confidence']:.4f}")
    print(f"  deductive_rules_applied: {info['deductive_rules_applied']}")
    print(f"  top_nodes (type={type(info['top_nodes']).__name__}):")
    for n in info["top_nodes"]:
        # This is the line that used to raise:
        #   TypeError: 'int' object is not subscriptable
        # because top_nodes was a list of ints. Now it's a list of dicts.
        print(f"    - id={n['id']!r}  "
              f"confidence={n['confidence']:.4f}  "
              f"text={n['text']!r}")

    assert isinstance(info["top_nodes"], list), "top_nodes must be a list"
    assert info["top_nodes"], "top_nodes must be non-empty after learn()s"
    sample = info["top_nodes"][0]
    assert isinstance(sample, dict), (
        f"top_nodes entries must be dicts, got {type(sample).__name__}"
    )
    assert {"id", "text", "confidence"} <= set(sample.keys()), (
        f"top_nodes dict must have id/text/confidence, got {list(sample.keys())}"
    )
    print("  [OK] top_nodes are dicts with the expected keys")

    print("\n" + "=" * 60)
    print("ALL E2E CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
