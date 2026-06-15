"""AGNN Multi-hop Traversal Benchmark.

Research question: Can AGNNGraph.traverse() answer multi-hop questions
that were never explicitly taught?

Example: If the graph knows:
    Sukarno --[CATEGORICAL/adalah]--> Presiden_Pertama
    Presiden_Pertama --[CATEGORICAL/lahir_di]--> Blitar
Can traverse from "Sukarno" produce a reasoning chain that reaches
"Blitar" — even though no node directly says "Sukarno lahir di Blitar"?

Benchmark design:
    1. Build a synthetic knowledge graph across 3 domains (Sejarah Indonesia,
       Biologi, Geografi) using numpy random embeddings + typed edges.
    2. Define 15+ multi-hop test cases with expected reasoning chains.
    3. Run traverse() under 3 conditions:
       - Baseline: raw random embeddings (no message passing)
       - +MessagePassing: after graph.message_pass_all()
       - +SpreadActivation: after graph.spread_activation() from seed node
    4. Score by node_recall: how many expected nodes appear in the chain.

Constraints:
    - No GPU, no real model, no internet — pure numpy + AGNNGraph
    - Run directly: python self-ai/tests/benchmark_agnn_traversal.py
    - Seed node selection uses simple keyword matching
    - Random numpy embeddings with seed=42 for reproducibility
"""

from __future__ import annotations

import sys
import os
import copy
import time

# Must set path before any agnn imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np

from agnn.graph import (
    AGNNGraph, AGNNNode, TypedEdge, ReasoningChain,
    NodeType, RelationType, EdgeRole,
)


# ══════════════════════════════════════════════════════════════
#  Knowledge Graph Construction
# ══════════════════════════════════════════════════════════════

def build_knowledge_graph() -> AGNNGraph:
    """Build a synthetic knowledge graph covering 3 domains.

    Domains:
        1. Sejarah Indonesia — presidents, birthplaces, events
        2. Biologi — animals, classifications, food chains
        3. Geografi — cities, countries, regions

    The graph uses typed edges with CATEGORICAL, SPATIAL, TEMPORAL,
    CAUSAL, and FUNCTIONAL relations to reflect the real semantics
    of each connection.
    """
    g = AGNNGraph(embedding_dim=64)
    np.random.seed(42)

    # ── Domain 1: Sejarah Indonesia ──
    sejarah_nodes = [
        ("Sukarno", "Sukarno", NodeType.ENTITY, 0.95),
        ("Soeharto", "Soeharto", NodeType.ENTITY, 0.95),
        ("Habibie", "Habibie", NodeType.ENTITY, 0.90),
        ("Presiden_Pertama", "Presiden Pertama", NodeType.CONCEPT, 0.90),
        ("Presiden_Kedua", "Presiden Kedua", NodeType.CONCEPT, 0.90),
        ("Presiden_Ketiga", "Presiden Ketiga", NodeType.CONCEPT, 0.88),
        ("Blitar", "Blitar", NodeType.ENTITY, 0.85),
        ("Kemusuk", "Kemusuk", NodeType.ENTITY, 0.85),
        ("Parepare", "Parepare", NodeType.ENTITY, 0.85),
        ("Proklamasi", "Proklamasi Kemerdekaan", NodeType.EVENT if hasattr(NodeType, 'EVENT') else NodeType.CONCEPT, 0.92),
        ("1945", "1945", NodeType.QUANTITY, 0.90),
    ]
    for nid, label, ntype, conf in sejarah_nodes:
        g.add_node(AGNNNode(id=nid, label=label, node_type=ntype, confidence=conf))

    sejarah_edges = [
        ("Sukarno", "Presiden_Pertama", RelationType.CATEGORICAL, 0.95, "adalah"),
        ("Presiden_Pertama", "Blitar", RelationType.SPATIAL, 0.90, "lahir_di"),
        ("Soeharto", "Presiden_Kedua", RelationType.CATEGORICAL, 0.95, "adalah"),
        ("Presiden_Kedua", "Kemusuk", RelationType.SPATIAL, 0.90, "lahir_di"),
        ("Habibie", "Presiden_Ketiga", RelationType.CATEGORICAL, 0.95, "adalah"),
        ("Presiden_Ketiga", "Parepare", RelationType.SPATIAL, 0.90, "lahir_di"),
        ("Sukarno", "Proklamasi", RelationType.CAUSAL, 0.92, "memimpin"),
        ("Proklamasi", "1945", RelationType.TEMPORAL, 0.95, "terjadi_pada"),
    ]
    for src, tgt, rel, conf, ctx in sejarah_edges:
        g.add_edge(TypedEdge(src, tgt, rel, confidence=conf, context=ctx))

    # ── Domain 2: Biologi ──
    biologi_nodes = [
        ("Harimau", "Harimau", NodeType.ENTITY, 0.92),
        ("Singa", "Singa", NodeType.ENTITY, 0.92),
        ("Elang", "Elang", NodeType.ENTITY, 0.90),
        ("Karnivora", "Karnivora", NodeType.CONCEPT, 0.88),
        ("Mamalia", "Mamalia", NodeType.CONCEPT, 0.88),
        ("Daging", "Daging", NodeType.ENTITY, 0.85),
        ("Vertebrata", "Vertebrata", NodeType.CONCEPT, 0.86),
    ]
    for nid, label, ntype, conf in biologi_nodes:
        g.add_node(AGNNNode(id=nid, label=label, node_type=ntype, confidence=conf))

    biologi_edges = [
        ("Harimau", "Karnivora", RelationType.CATEGORICAL, 0.95, "adalah"),
        ("Singa", "Karnivora", RelationType.CATEGORICAL, 0.95, "adalah"),
        ("Elang", "Karnivora", RelationType.CATEGORICAL, 0.90, "adalah"),
        ("Karnivora", "Daging", RelationType.CAUSAL, 0.88, "memakan"),
        ("Karnivora", "Mamalia", RelationType.CATEGORICAL, 0.85, "bagian_dari"),
        ("Mamalia", "Vertebrata", RelationType.CATEGORICAL, 0.85, "bagian_dari"),
    ]
    for src, tgt, rel, conf, ctx in biologi_edges:
        g.add_edge(TypedEdge(src, tgt, rel, confidence=conf, context=ctx))

    # ── Domain 3: Geografi ──
    geografi_nodes = [
        ("Jakarta", "Jakarta", NodeType.ENTITY, 0.93),
        ("Bali", "Bali", NodeType.ENTITY, 0.90),
        ("Ibukota", "Ibukota", NodeType.CONCEPT, 0.88),
        ("Indonesia", "Indonesia", NodeType.ENTITY, 0.95),
        ("Asia_Tenggara", "Asia Tenggara", NodeType.CONCEPT, 0.90),
        ("Pulau_Jawa", "Pulau Jawa", NodeType.CONCEPT, 0.88),
        ("Pulau_Bali", "Pulau Bali", NodeType.CONCEPT, 0.87),
    ]
    for nid, label, ntype, conf in geografi_nodes:
        g.add_node(AGNNNode(id=nid, label=label, node_type=ntype, confidence=conf))

    geografi_edges = [
        ("Jakarta", "Ibukota", RelationType.CATEGORICAL, 0.95, "adalah"),
        ("Ibukota", "Indonesia", RelationType.SPATIAL, 0.92, "bagian_dari"),
        ("Indonesia", "Asia_Tenggara", RelationType.SPATIAL, 0.90, "bagian_dari"),
        ("Jakarta", "Pulau_Jawa", RelationType.SPATIAL, 0.88, "terletak_di"),
        ("Bali", "Pulau_Bali", RelationType.SPATIAL, 0.90, "terletak_di"),
        ("Pulau_Bali", "Indonesia", RelationType.SPATIAL, 0.88, "bagian_dari"),
        ("Pulau_Jawa", "Indonesia", RelationType.SPATIAL, 0.90, "bagian_dari"),
    ]
    for src, tgt, rel, conf, ctx in geografi_edges:
        g.add_edge(TypedEdge(src, tgt, rel, confidence=conf, context=ctx))

    return g


# ══════════════════════════════════════════════════════════════
#  Test Cases
# ══════════════════════════════════════════════════════════════

def get_test_cases() -> list[dict]:
    """Define multi-hop test cases across all 3 domains.

    Each test case:
        - question: the query text (used for seed node selection)
        - expected_nodes: nodes that MUST appear in the reasoning chain
        - min_hops: minimum hops needed to connect all expected nodes
        - domain: which domain this test belongs to

    Hops definition:
        1-hop = direct edge from seed (baseline)
        2-hop = needs to traverse 1 intermediate node
        3-hop = needs to traverse 2 intermediate nodes
    """
    return [
        # ── Sejarah Indonesia: 1-hop (baseline) ──
        {
            "question": "Sukarno adalah apa?",
            "expected_nodes": ["Sukarno", "Presiden_Pertama"],
            "min_hops": 1,
            "domain": "Sejarah Indonesia",
        },
        {
            "question": "Soeharto adalah apa?",
            "expected_nodes": ["Soeharto", "Presiden_Kedua"],
            "min_hops": 1,
            "domain": "Sejarah Indonesia",
        },
        # ── Sejarah Indonesia: 2-hop ──
        {
            "question": "Sukarno lahir di mana?",
            "expected_nodes": ["Sukarno", "Presiden_Pertama", "Blitar"],
            "min_hops": 2,
            "domain": "Sejarah Indonesia",
        },
        {
            "question": "Soeharto lahir di mana?",
            "expected_nodes": ["Soeharto", "Presiden_Kedua", "Kemusuk"],
            "min_hops": 2,
            "domain": "Sejarah Indonesia",
        },
        {
            "question": "Habibie lahir di mana?",
            "expected_nodes": ["Habibie", "Presiden_Ketiga", "Parepare"],
            "min_hops": 2,
            "domain": "Sejarah Indonesia",
        },
        # ── Sejarah Indonesia: 3-hop ──
        {
            "question": "Kapan Sukarno memimpin proklamasi?",
            "expected_nodes": ["Sukarno", "Proklamasi", "1945"],
            "min_hops": 2,
            "domain": "Sejarah Indonesia",
        },
        # ── Biologi: 1-hop (baseline) ──
        {
            "question": "Harimau adalah apa?",
            "expected_nodes": ["Harimau", "Karnivora"],
            "min_hops": 1,
            "domain": "Biologi",
        },
        {
            "question": "Singa adalah apa?",
            "expected_nodes": ["Singa", "Karnivora"],
            "min_hops": 1,
            "domain": "Biologi",
        },
        # ── Biologi: 2-hop ──
        {
            "question": "Apa yang dimakan harimau?",
            "expected_nodes": ["Harimau", "Karnivora", "Daging"],
            "min_hops": 2,
            "domain": "Biologi",
        },
        {
            "question": "Singa termasuk mamalia?",
            "expected_nodes": ["Singa", "Karnivora", "Mamalia"],
            "min_hops": 2,
            "domain": "Biologi",
        },
        # ── Biologi: 3-hop ──
        {
            "question": "Harimau termasuk vertebrata?",
            "expected_nodes": ["Harimau", "Karnivora", "Mamalia", "Vertebrata"],
            "min_hops": 3,
            "domain": "Biologi",
        },
        # ── Geografi: 1-hop (baseline) ──
        {
            "question": "Jakarta adalah apa?",
            "expected_nodes": ["Jakarta", "Ibukota"],
            "min_hops": 1,
            "domain": "Geografi",
        },
        # ── Geografi: 2-hop ──
        {
            "question": "Jakarta bagian dari negara mana?",
            "expected_nodes": ["Jakarta", "Ibukota", "Indonesia"],
            "min_hops": 2,
            "domain": "Geografi",
        },
        {
            "question": "Bali termasuk Indonesia?",
            "expected_nodes": ["Bali", "Pulau_Bali", "Indonesia"],
            "min_hops": 2,
            "domain": "Geografi",
        },
        # ── Geografi: 3-hop ──
        {
            "question": "Jakarta berada di wilayah apa?",
            "expected_nodes": ["Jakarta", "Ibukota", "Indonesia", "Asia_Tenggara"],
            "min_hops": 3,
            "domain": "Geografi",
        },
        {
            "question": "Jakarta terletak di pulau mana?",
            "expected_nodes": ["Jakarta", "Pulau_Jawa"],
            "min_hops": 1,
            "domain": "Geografi",
        },
    ]


# ══════════════════════════════════════════════════════════════
#  Seed Node Selection (Keyword Matching)
# ══════════════════════════════════════════════════════════════

def find_seed_node(graph: AGNNGraph, question: str) -> str | None:
    """Find the best seed node for a question using keyword matching.

    Strategy:
        1. Tokenize the question into lowercase words
        2. Check if any word matches a node label (case-insensitive)
        3. Among matches, pick the one with highest confidence
        4. If no match, try substring matching (is any node label
           a substring of the question?)
    """
    words = question.lower().replace("?", "").replace(".", "").split()

    best_node_id = None
    best_conf = -1.0

    for node in graph._nodes.values():
        label_lower = node.label.lower()
        label_words = label_lower.split()

        # Check if any question word matches any label word
        for w in words:
            if w in label_words or label_lower == w:
                if node.confidence > best_conf:
                    best_node_id = node.id
                    best_conf = node.confidence
                break

    # Fallback: substring matching
    if best_node_id is None:
        question_lower = question.lower()
        for node in graph._nodes.values():
            label_lower = node.label.lower()
            if label_lower in question_lower or question_lower in label_lower:
                if node.confidence > best_conf:
                    best_node_id = node.id
                    best_conf = node.confidence

    return best_node_id


# ══════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════

def score_traversal(chain: ReasoningChain | None,
                    expected_nodes: list[str]) -> dict:
    """Score a traversal result against expected nodes.

    Returns:
        dict with:
            - node_recall: fraction of expected nodes found in chain
            - chain_length: number of steps in the chain (0 if None)
            - traversal_success: True if ALL expected nodes are in chain
            - found_nodes: list of expected nodes that were found
            - missing_nodes: list of expected nodes that were missing
    """
    if chain is None:
        return {
            "node_recall": 0.0,
            "chain_length": 0,
            "traversal_success": False,
            "found_nodes": [],
            "missing_nodes": list(expected_nodes),
        }

    # chain.node_ids contains all visited node IDs
    visited = set(chain.node_ids)

    found = [n for n in expected_nodes if n in visited]
    missing = [n for n in expected_nodes if n not in visited]

    recall = len(found) / len(expected_nodes) if expected_nodes else 0.0

    return {
        "node_recall": recall,
        "chain_length": len(chain.steps),
        "traversal_success": len(found) == len(expected_nodes),
        "found_nodes": found,
        "missing_nodes": missing,
    }


# ══════════════════════════════════════════════════════════════
#  Benchmark Runner
# ══════════════════════════════════════════════════════════════

def run_benchmark():
    """Run the full multi-hop traversal benchmark.

    Tests three conditions:
        1. Baseline: traverse with raw random embeddings
        2. +MessagePassing: after graph.message_pass_all()
        3. +SpreadActivation: traverse using spread_activation to guide path
    """
    print("=" * 60)
    print("  AGNN Multi-hop Traversal Benchmark")
    print("=" * 60)
    print()

    # Build the knowledge graph
    print("[1/4] Building knowledge graph...")
    graph = build_knowledge_graph()
    print(f"      Nodes: {graph.node_count()}, Edges: {graph.edge_count()}")
    print()

    # Get test cases
    test_cases = get_test_cases()
    print(f"[2/4] Loaded {len(test_cases)} test cases")
    print()

    # ── Condition 1: Baseline (raw random embeddings) ──
    print("[3/4] Running benchmark conditions...")
    print()
    print("  Condition 1: Baseline (raw random embeddings)")
    baseline_results = []
    for tc in test_cases:
        seed_id = find_seed_node(graph, tc["question"])
        if seed_id is None:
            baseline_results.append(score_traversal(None, tc["expected_nodes"]))
            continue
        chain = graph.traverse(seed_id, max_hops=3)
        result = score_traversal(chain, tc["expected_nodes"])
        result["seed_id"] = seed_id
        result["chain"] = chain
        baseline_results.append(result)

    # ── Condition 2: After Message Passing ──
    # Deep copy graph so message passing doesn't affect baseline
    print("  Condition 2: After Message Passing")
    graph_mp = build_knowledge_graph()  # fresh graph with same seed
    np.random.seed(42)  # ensure same random embeddings
    graph_mp.message_pass_all(damping=0.5, iterations=3)

    mp_results = []
    for tc in test_cases:
        seed_id = find_seed_node(graph_mp, tc["question"])
        if seed_id is None:
            mp_results.append(score_traversal(None, tc["expected_nodes"]))
            continue
        chain = graph_mp.traverse(seed_id, max_hops=3)
        result = score_traversal(chain, tc["expected_nodes"])
        result["seed_id"] = seed_id
        result["chain"] = chain
        mp_results.append(result)

    # ── Condition 3: After Spread Activation ──
    # Fresh graph, then spread activation from seed before traversing
    print("  Condition 3: After Spread Activation")
    graph_sa = build_knowledge_graph()
    np.random.seed(42)
    # Also do message passing first, then spread activation
    graph_sa.message_pass_all(damping=0.5, iterations=3)

    sa_results = []
    for tc in test_cases:
        seed_id = find_seed_node(graph_sa, tc["question"])
        if seed_id is None:
            sa_results.append(score_traversal(None, tc["expected_nodes"]))
            continue
        # Spread activation from seed node
        activation = graph_sa.spread_activation([seed_id], steps=3)
        # Use the activation map to boost confidence of highly activated nodes
        # This should improve traversal because higher-confidence nodes
        # are preferred by the BFS priority queue
        for nid, act in activation.items():
            node = graph_sa.get_node(nid)
            if node is not None:
                # Blend original confidence with activation
                node.confidence = min(1.0, 0.5 * node.confidence + 0.5 * act)

        chain = graph_sa.traverse(seed_id, max_hops=3)
        result = score_traversal(chain, tc["expected_nodes"])
        result["seed_id"] = seed_id
        result["chain"] = chain
        sa_results.append(result)

    # ── Print Results ──
    print()
    print("[4/4] Generating report...")
    print()
    print_results_table(test_cases, baseline_results, mp_results, sa_results)


def print_results_table(test_cases, baseline_results, mp_results, sa_results):
    """Print the benchmark results in a formatted table."""

    # Group by domain
    domains = {}
    for i, tc in enumerate(test_cases):
        domain = tc["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(i)

    for domain, indices in domains.items():
        print(f"\n=== AGNN Multi-hop Traversal Benchmark ===")
        print(f"Domain: {domain}")
        print()

        # Table header
        header = (
            f"{'Question':<35} {'Hops':>5} "
            f"{'Baseline':>16} {'+MsgPass':>16} {'+SpreadAct':>16}"
        )
        separator = "-" * len(header)

        print(separator)
        print(header)
        print(separator)

        for i in indices:
            tc = test_cases[i]
            b = baseline_results[i]
            m = mp_results[i]
            s = sa_results[i]

            question_short = tc["question"][:33] + ".." if len(tc["question"]) > 35 else tc["question"]

            baseline_str = f"recall={b['node_recall']:.2f}"
            mp_str = f"recall={m['node_recall']:.2f}"
            sa_str = f"recall={s['node_recall']:.2f}"

            print(
                f"{question_short:<35} {tc['min_hops']:>5} "
                f"{baseline_str:>16} {mp_str:>16} {sa_str:>16}"
            )

        print(separator)

        # Domain summary
        domain_baseline = [baseline_results[i]["node_recall"] for i in indices]
        domain_mp = [mp_results[i]["node_recall"] for i in indices]
        domain_sa = [sa_results[i]["node_recall"] for i in indices]

        domain_baseline_success = sum(1 for i in indices if baseline_results[i]["traversal_success"])
        domain_mp_success = sum(1 for i in indices if mp_results[i]["traversal_success"])
        domain_sa_success = sum(1 for i in indices if sa_results[i]["traversal_success"])

        print(f"\n  Domain Summary ({domain}):")
        print(f"    Baseline avg recall:           {np.mean(domain_baseline):.2f}")
        print(f"    After msg pass avg recall:     {np.mean(domain_mp):.2f}  ({np.mean(domain_mp) - np.mean(domain_baseline):+.2f})")
        print(f"    After spread act avg recall:   {np.mean(domain_sa):.2f}  ({np.mean(domain_sa) - np.mean(domain_baseline):+.2f})")
        print(f"    Baseline traversal success:    {domain_baseline_success}/{len(indices)}")
        print(f"    Msg pass traversal success:    {domain_mp_success}/{len(indices)}")
        print(f"    Spread act traversal success:  {domain_sa_success}/{len(indices)}")
        print()

    # ── Overall Summary ──
    all_baseline = [r["node_recall"] for r in baseline_results]
    all_mp = [r["node_recall"] for r in mp_results]
    all_sa = [r["node_recall"] for r in sa_results]

    baseline_success = sum(1 for r in baseline_results if r["traversal_success"])
    mp_success = sum(1 for r in mp_results if r["traversal_success"])
    sa_success = sum(1 for r in sa_results if r["traversal_success"])

    total = len(test_cases)

    print("=" * 60)
    print("  OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Baseline avg recall:              {np.mean(all_baseline):.2f}")
    print(f"  After message passing avg recall:  {np.mean(all_mp):.2f}  ({np.mean(all_mp) - np.mean(all_baseline):+.2f})")
    print(f"  After spread activation avg recall:{np.mean(all_sa):.2f}  ({np.mean(all_sa) - np.mean(all_baseline):+.2f})")
    print()
    print(f"  % questions where traverse found all expected nodes:")
    print(f"    Baseline:             {baseline_success}/{total} ({100*baseline_success/total:.0f}%)")
    print(f"    After message passing:{mp_success}/{total} ({100*mp_success/total:.0f}%)")
    print(f"    After spread activ.:  {sa_success}/{total} ({100*sa_success/total:.0f}%)")
    print()

    # ── Hop-level breakdown ──
    print("  Recall by hop depth:")
    for hop in [1, 2, 3]:
        hop_indices = [i for i, tc in enumerate(test_cases) if tc["min_hops"] == hop]
        if not hop_indices:
            continue
        hop_baseline = [baseline_results[i]["node_recall"] for i in hop_indices]
        hop_mp = [mp_results[i]["node_recall"] for i in hop_indices]
        hop_sa = [sa_results[i]["node_recall"] for i in hop_indices]
        print(f"    {hop}-hop ({len(hop_indices)} questions):")
        print(f"      Baseline:       {np.mean(hop_baseline):.2f}")
        print(f"      +MsgPass:       {np.mean(hop_mp):.2f}  ({np.mean(hop_mp) - np.mean(hop_baseline):+.2f})")
        print(f"      +SpreadActiv.:  {np.mean(hop_sa):.2f}  ({np.mean(hop_sa) - np.mean(hop_baseline):+.2f})")
    print()

    # ── Failure Analysis ──
    print("  FAILURE ANALYSIS:")
    print("  Cases where ALL conditions failed (0.00 recall):")
    all_failed = []
    for i, tc in enumerate(test_cases):
        if baseline_results[i]["node_recall"] == 0 and mp_results[i]["node_recall"] == 0 and sa_results[i]["node_recall"] == 0:
            all_failed.append(i)
    if all_failed:
        for i in all_failed:
            tc = test_cases[i]
            b = baseline_results[i]
            print(f"    - {tc['question']} (hops={tc['min_hops']})")
            print(f"      Seed: {b.get('seed_id', 'N/A')}, Missing: {b['missing_nodes']}")
    else:
        print("    (none — all questions had at least partial recall)")

    print()
    print("  Cases that improved MOST with message passing:")
    improvements = []
    for i, tc in enumerate(test_cases):
        delta = mp_results[i]["node_recall"] - baseline_results[i]["node_recall"]
        improvements.append((delta, i))
    improvements.sort(reverse=True)
    for delta, i in improvements[:5]:
        if delta > 0:
            tc = test_cases[i]
            print(f"    - {tc['question']} (hops={tc['min_hops']}): {baseline_results[i]['node_recall']:.2f} -> {mp_results[i]['node_recall']:.2f} ({delta:+.2f})")

    print()
    print("  Cases that improved MOST with spread activation:")
    improvements_sa = []
    for i, tc in enumerate(test_cases):
        delta = sa_results[i]["node_recall"] - baseline_results[i]["node_recall"]
        improvements_sa.append((delta, i))
    improvements_sa.sort(reverse=True)
    for delta, i in improvements_sa[:5]:
        if delta > 0:
            tc = test_cases[i]
            print(f"    - {tc['question']} (hops={tc['min_hops']}): {baseline_results[i]['node_recall']:.2f} -> {sa_results[i]['node_recall']:.2f} ({delta:+.2f})")

    print()
    print("  Per-question chain details:")
    for i, tc in enumerate(test_cases):
        b = baseline_results[i]
        if b.get("chain"):
            print(f"    [{tc['question']}]")
            print(f"      Baseline chain: {b['chain'].verbalize()}")
            print(f"      Nodes visited: {b['chain'].node_ids}")
            print(f"      Expected: {tc['expected_nodes']}, Found: {b['found_nodes']}, Missing: {b['missing_nodes']}")
        else:
            print(f"    [{tc['question']}]")
            print(f"      Baseline: NO CHAIN FOUND")
            print(f"      Expected: {tc['expected_nodes']}")

    print()
    print("=" * 60)
    print("  ANALYSIS")
    print("=" * 60)
    print()
    analyze_results(test_cases, baseline_results, mp_results, sa_results)


def analyze_results(test_cases, baseline_results, mp_results, sa_results):
    """Generate analysis of benchmark results."""
    all_baseline = [r["node_recall"] for r in baseline_results]
    all_mp = [r["node_recall"] for r in mp_results]
    all_sa = [r["node_recall"] for r in sa_results]

    # 1. Does message passing help?
    mp_delta = np.mean(all_mp) - np.mean(all_baseline)
    if mp_delta > 0.05:
        print("1. Does message passing help traversal?")
        print(f"   YES. Message passing improved average recall by {mp_delta:+.2f}.")
        print("   Message passing updates node embeddings by aggregating")
        print("   information from neighbors. After message passing, nodes")
        print("   that are structurally close in the graph have more similar")
        print("   embeddings. However, traverse() uses BFS with confidence-weighted")
        print("   priority, not embedding similarity, so the effect is indirect:")
        print("   message passing does NOT change edge confidences or graph topology,")
        print("   which are what traverse() actually follows.")
    elif mp_delta > 0:
        print("1. Does message passing help traversal?")
        print(f"   MARGINALLY. Message passing improved average recall by {mp_delta:+.2f}.")
        print("   The improvement is minimal because traverse() uses BFS with")
        print("   confidence-weighted priority — it follows graph topology and")
        print("   edge confidences, not embedding similarity. Message passing")
        print("   updates embeddings but does not modify the graph structure or")
        print("   edge weights that the traversal algorithm actually uses.")
    else:
        print("1. Does message passing help traversal?")
        print(f"   NO. Message passing did not improve recall ({mp_delta:+.2f}).")
        print("   This is expected because traverse() follows graph topology")
        print("   (BFS on edges) and does not use embeddings for path selection.")
        print("   Message passing only changes node embeddings, which are not")
        print("   consulted during traversal — the algorithm uses edge confidence")
        print("   and relation weights instead.")

    print()

    # 2. Failure patterns
    print("2. What are the most common failure patterns?")
    # Count failures by hop depth
    hop_failures = {}
    for i, tc in enumerate(test_cases):
        hop = tc["min_hops"]
        if hop not in hop_failures:
            hop_failures[hop] = {"total": 0, "baseline_fail": 0, "sa_fail": 0}
        hop_failures[hop]["total"] += 1
        if not baseline_results[i]["traversal_success"]:
            hop_failures[hop]["baseline_fail"] += 1
        if not sa_results[i]["traversal_success"]:
            hop_failures[hop]["sa_fail"] += 1

    for hop in sorted(hop_failures.keys()):
        data = hop_failures[hop]
        print(f"   {hop}-hop: {data['baseline_fail']}/{data['total']} failures (baseline), "
              f"{data['sa_fail']}/{data['total']} failures (+spread activation)")

    # Analyze WHY failures happen
    missing_node_counts = {}
    for i, tc in enumerate(test_cases):
        for missing in baseline_results[i]["missing_nodes"]:
            missing_node_counts[missing] = missing_node_counts.get(missing, 0) + 1

    if missing_node_counts:
        print("   Most frequently missing nodes:")
        for node, count in sorted(missing_node_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"     - {node}: missing in {count} test cases")

    print()

    # 3. Concrete recommendation
    print("3. Single most impactful recommendation to improve recall:")
    # Analyze: traverse() does BFS on outgoing edges. It picks the
    # highest-confidence path. If a multi-hop path exists but the
    # intermediate node has outgoing edges going in a "wrong" direction,
    # the greedy BFS will miss the target.
    #
    # The real issue: traverse() is greedy BFS. It doesn't explore ALL
    # paths — it picks the best-scoring one. For multi-hop reasoning,
    # we need either:
    # (a) Multiple chain generation (explore top-K paths)
    # (b) Guided traversal using activation scores
    # (c) Bidirectional search (from both seed and target)

    baseline_success = sum(1 for r in baseline_results if r["traversal_success"])
    total = len(test_cases)

    if baseline_success < total:
        print("   IMPLEMENT ACTIVITY-GUIDED TRAVERSAL.")
        print("   Current traverse() uses greedy BFS that follows only the")
        print("   highest-confidence path. For multi-hop questions, the correct")
        print("   path may not be the highest-confidence one at each step.")
        print("   Instead, use spread_activation() to compute relevance scores")
        print("   for ALL nodes, then use those scores to guide the BFS priority.")
        print("   Specifically: replace the confidence-based priority with")
        print("   activation_score * edge_confidence, so that nodes with high")
        print("   activation (relevance to the query) are explored first.")
        print()
        print("   This single change would make traverse() query-aware instead")
        print("   of purely topology-driven, which is the core gap exposed by")
        print("   this benchmark.")
    else:
        print("   (All test cases pass — no improvement needed for this benchmark.)")

    print()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_benchmark()
