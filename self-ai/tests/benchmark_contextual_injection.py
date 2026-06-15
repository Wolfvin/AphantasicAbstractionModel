#!/usr/bin/env python3
"""Benchmark: end-to-end contextual injection via CompositionLayer.answer().

v41: This benchmark tests the new automatic retrieve+inject flow in
CompositionLayer.answer(). The flow is:

    question
      → UnderstandingGraph.retrieve(question, top_k=5)
      → list[(node, score)]  (kosong jika graph belum punya nodes)
      → UnconsciousInjector.active(nodes)  (skip jika kosong)
      → model.generate(question)
      → answer

Three conditions are tested:
  1. BASELINE — graph kosong, no injection (conscious-only path)
  2. POPULATED_NO_INJECTION — graph populated, injector not wired
     (answers use retrieve but no injection steering)
  3. POPULATED_WITH_INJECTION — graph populated + injector wired
     (full retrieve+inject pipeline)

Metrics (same as benchmark_l2_multi_injection.py):
  1. keyword_hit_rate  — does the correct answer keyword appear in output? (0.0–1.0)
  2. answer_alignment  — cosine sim of output embedding to correct answer text embedding
  3. semantic_score    — weighted combo: 0.6 * keyword_hit_rate + 0.4 * answer_alignment

Run:
  cd self-ai/src
  python ../tests/benchmark_contextual_injection.py

  # Or from project root:
  python self-ai/tests/benchmark_contextual_injection.py
"""

import os, sys, logging, json, statistics, uuid, re, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('benchmark_contextual')
logger.setLevel(logging.INFO)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dependencies', 'models'))
QWEN_PATH = os.path.join(MODELS_DIR, 'Qwen3-0.6B')
BGE_PATH  = os.path.join(MODELS_DIR, 'bge-m3')

# ── Test cases ────────────────────────────────────────────────────────────────
# Same 4 questions as benchmark_l2_multi_injection.py
TEST_CASES = [
    {
        "question": "Sebutkan satu hewan yang tidak termasuk dalam kelompok: kucing, anjing, ikan, harimau.",
        "correct_answer": "harimau tidak termasuk dalam kelompok hewan peliharaan",
        "correct_keywords": ["harimau"],
    },
    {
        "question": "Berapa hasil dari 50 dikurangi jumlah yang hilang jika ada 15 yang hilang?",
        "correct_answer": "35, karena 50 dikurangi 15 sama dengan 35",
        "correct_keywords": ["35"],
    },
    {
        "question": "Siapa yang tidak lulus jika semua lulus kecuali Budi?",
        "correct_answer": "Budi adalah yang tidak lulus",
        "correct_keywords": ["budi"],
    },
    {
        "question": "Sebutkan bilangan yang bukan bilangan prima dari: 2, 3, 4, 5.",
        "correct_answer": "4 bukan bilangan prima karena dapat dibagi 2",
        "correct_keywords": ["4"],
    },
]

GEN_KWARGS = dict(max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
N_RUNS = 4


# ── Metric helpers ────────────────────────────────────────────────────────────

def keyword_hit(output: str, keywords: list) -> float:
    """Return 1.0 if any keyword found in output (case-insensitive), else 0.0."""
    out_lower = output.lower()
    return 1.0 if any(kw.lower() in out_lower for kw in keywords) else 0.0


def cosine_sim(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    return float((a / (a.norm() + 1e-8)) @ (b / (b.norm() + 1e-8)))


def semantic_score(hit_rate: float, alignment: float) -> float:
    return 0.6 * hit_rate + 0.4 * alignment


def generate(model, tokenizer, prompt, **kwargs):
    inputs = tokenizer(prompt, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(
        out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()


# ── Graph population helper ──────────────────────────────────────────────────

def populate_graph(graph, bge):
    """Populate UnderstandingGraph with nodes from training_pairs_dataset.

    Each experience text becomes one UnderstandingNode with confidence 0.7.
    The node's condition_embedding is computed via bge-m3.

    Returns:
        Number of nodes added.
    """
    from unconscious.training_pairs_dataset import get_training_pairs
    from derivation.understanding_builder import UnderstandingNode

    pairs = get_training_pairs()
    count = 0
    for i, (experience_text, _output_text) in enumerate(pairs):
        node_id = f"bench_ctx_{i:04d}"
        node = UnderstandingNode(
            id=node_id,
            name=experience_text[:40],
            concept=experience_text,
            abstraction=experience_text,
            confidence=0.7,
            source='benchmark_contextual',
        )
        # Compute embedding before adding (so add_node doesn't need to)
        emb = bge.encode([experience_text], normalize_embeddings=True, show_progress_bar=False)[0]
        node.condition_embedding = emb.tolist()
        graph.add_node(node)
        count += 1

    logger.info("Populated graph with %d nodes from training_pairs_dataset", count)
    return count


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_condition(label, model, tokenizer, injector, composition_layer,
                  bge, n_runs, use_graph=False, use_injection=False):
    """
    Run all TEST_CASES under one condition.

    Args:
        label: Condition name for logging.
        model: Qwen3 model instance.
        tokenizer: Qwen3 tokenizer instance.
        injector: UnconsciousInjector instance (or None).
        composition_layer: CompositionLayer instance with graph/injector wired.
        bge: bge-m3 SentenceTransformer instance.
        n_runs: Number of runs per test case.
        use_graph: If True, wire graph to composition layer.
        use_injection: If True, wire injector to composition layer.

    Returns:
        List of per-case result dicts.
    """
    # Pre-embed all correct answers
    correct_embs = bge.encode(
        [tc["correct_answer"] for tc in TEST_CASES],
        normalize_embeddings=True, show_progress_bar=False
    )

    results = []
    for tc, correct_emb in zip(TEST_CASES, correct_embs):
        outputs = []
        for _ in range(n_runs):
            if use_graph and use_injection:
                # Full end-to-end: answer() with graph + injector
                text = composition_layer.answer(tc["question"], max_new_tokens=80)
            elif use_graph:
                # Graph populated but no injector — answer() without injection
                text = composition_layer.answer(tc["question"], max_new_tokens=80)
            else:
                # Baseline — no graph, no injector, direct generate
                text = generate(model, tokenizer, tc["question"], **GEN_KWARGS)

            # Fallback: if answer() returns empty, retry once
            if not text or len(text.strip()) < 3:
                if use_graph:
                    text = composition_layer.answer(tc["question"], max_new_tokens=128)
                else:
                    text = generate(model, tokenizer, tc["question"],
                                    max_new_tokens=128, do_sample=True,
                                    temperature=0.7, top_p=0.9)
            outputs.append(text)

        # Embed outputs
        output_embs = bge.encode(outputs, normalize_embeddings=True, show_progress_bar=False)

        # Per-run metrics
        hits = [keyword_hit(o, tc["correct_keywords"]) for o in outputs]
        alignments = [cosine_sim(e, correct_emb) for e in output_embs]
        scores = [semantic_score(h, a) for h, a in zip(hits, alignments)]

        results.append({
            "question": tc["question"][:55],
            "keywords": tc["correct_keywords"],
            "outputs": outputs,
            "keyword_hit_rate": round(sum(hits) / n_runs, 4),
            "answer_alignment":  round(sum(alignments) / n_runs, 4),
            "semantic_score":    round(sum(scores) / n_runs, 4),
            "consistency":       round(1.0 - (statistics.stdev(scores) if len(scores) > 1 else 0), 4),
        })

    return results


def print_condition(label, results):
    logger.info("\n── %s ──", label)
    total_hit, total_align, total_score, total_cons = 0, 0, 0, 0
    for r in results:
        logger.info("  Q: %s", r["question"])
        logger.info("    kw=%s  hit=%.2f  align=%.4f  score=%.4f  cons=%.4f",
                    r["keywords"], r["keyword_hit_rate"],
                    r["answer_alignment"], r["semantic_score"], r["consistency"])
        logger.info("    sample: %s", r["outputs"][0][:90])
        total_hit   += r["keyword_hit_rate"]
        total_align += r["answer_alignment"]
        total_score += r["semantic_score"]
        total_cons  += r["consistency"]
    n = len(results)
    means = dict(
        hit_rate=round(total_hit/n, 4),
        alignment=round(total_align/n, 4),
        score=round(total_score/n, 4),
        consistency=round(total_cons/n, 4),
    )
    logger.info("  MEAN  hit=%.4f  align=%.4f  score=%.4f  cons=%.4f",
                means["hit_rate"], means["alignment"], means["score"], means["consistency"])
    return means


def run_random_vector_condition(label, model, tokenizer, injector, bge, n_runs, seed=42):
    """Condition 4: inject random normalized vector — no graph, no semantic content.

    This is the make-or-break ablation control. We bypass graph retrieval
    entirely and inject a random normalized 1024-dim vector directly into
    the injector's _experience_vector. The hook fires as normal — the only
    difference is the injected content is noise, not projected experience.

    If this scores similarly to POPULATED_WITH_INJECTION:
        → The effect is from hidden-state perturbation, not semantic content.
        → Graph-based memory claims must be reframed.

    If this scores similarly to BASELINE:
        → Semantic content of the projected vector matters.
        → Graph-based memory hypothesis is supported.

    Args:
        seed: RNG seed for reproducibility across runs.
    """
    correct_embs = bge.encode(
        [tc["correct_answer"] for tc in TEST_CASES],
        normalize_embeddings=True, show_progress_bar=False
    )

    # Build a random normalized vector with the same shape as a real injection
    # (1024-dim, float16, on model device) — same magnitude as projected vectors
    device = next(model.parameters()).device
    rng = torch.Generator()
    rng.manual_seed(seed)
    rand_vec = torch.randn(1024, generator=rng, dtype=torch.float16, device=device)
    rand_vec = rand_vec / (rand_vec.norm() + 1e-8)

    results = []
    for tc, correct_emb in zip(TEST_CASES, correct_embs):
        outputs = []
        for run_i in range(n_runs):
            # Each run uses a different seed so we're not just repeating one vector
            run_rng = torch.Generator()
            run_rng.manual_seed(seed + run_i)
            rv = torch.randn(1024, generator=run_rng, dtype=torch.float16, device=device)
            rv = rv / (rv.norm() + 1e-8)

            # Manually set the experience vector and register hook
            injector._experience_vector = rv
            injector._active = True
            try:
                layers = model.model.layers
                hook_handle = layers[injector.hook_layer_index].register_forward_hook(
                    injector._hook_fn
                )
                try:
                    text = generate(model, tokenizer, tc["question"], **GEN_KWARGS)
                finally:
                    hook_handle.remove()
            finally:
                injector._active = False
                injector._experience_vector = None

            if not text or len(text.strip()) < 3:
                text = generate(model, tokenizer, tc["question"],
                                max_new_tokens=128, do_sample=True,
                                temperature=0.7, top_p=0.9)
            outputs.append(text)

        output_embs = bge.encode(outputs, normalize_embeddings=True, show_progress_bar=False)
        hits = [keyword_hit(o, tc["correct_keywords"]) for o in outputs]
        alignments = [cosine_sim(e, correct_emb) for e in output_embs]
        scores = [semantic_score(h, a) for h, a in zip(hits, alignments)]

        results.append({
            "question": tc["question"][:55],
            "keywords": tc["correct_keywords"],
            "outputs": outputs,
            "keyword_hit_rate": round(sum(hits) / n_runs, 4),
            "answer_alignment":  round(sum(alignments) / n_runs, 4),
            "semantic_score":    round(sum(scores) / n_runs, 4),
            "consistency":       round(1.0 - (statistics.stdev(scores) if len(scores) > 1 else 0), 4),
        })

    return results


def main():
    # Prevent huggingface_hub from making network requests during benchmark
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger.info("Loading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, dtype=torch.float16)
    model.eval()

    logger.info("Loading bge-m3...")
    bge = SentenceTransformer(BGE_PATH)

    # ── Set up injector ───────────────────────────────────────────────────────
    from unconscious.injector import UnconsciousInjector
    from unconscious.projection_trainer import ProjectionTrainer

    injector = UnconsciousInjector(model)

    # Train/load projection
    weights_path = os.path.join(SRC_DIR, 'unconscious', 'projection_weights.pt')
    if not os.path.exists(weights_path):
        logger.info("Training projection...")
        from unconscious.training_pairs_dataset import get_training_pairs
        pairs = get_training_pairs()
        trainer = ProjectionTrainer(model, tokenizer, embedding_model=bge, hook_layer_index=14)
        trainer.train(pairs, epochs=80)
        trainer.load_into_injector(injector)
    else:
        logger.info("Loading pre-trained projection weights...")
        injector._try_load_trained_projection()

    # ── Set up CompositionLayer ───────────────────────────────────────────────
    from composition.layer import CompositionLayer

    # ── Condition 1: BASELINE (graph kosong, no injection) ───────────────────
    logger.info("\n=== Condition 1: BASELINE (empty graph, no injection) ===")
    comp_baseline = CompositionLayer()
    comp_baseline._model = model
    comp_baseline._tokenizer = tokenizer
    # No graph, no injector — plain generation
    baseline_results = run_condition(
        "BASELINE", model, tokenizer, injector, comp_baseline,
        bge, N_RUNS, use_graph=False, use_injection=False
    )
    baseline_m = print_condition("BASELINE", baseline_results)

    # ── Condition 2: POPULATED, NO INJECTION ─────────────────────────────────
    logger.info("\n=== Condition 2: POPULATED NO INJECTION ===")
    from derivation.understanding_builder import UnderstandingGraph
    graph_no_inj = UnderstandingGraph(embedding_model=bge)
    populate_graph(graph_no_inj, bge)

    comp_no_inj = CompositionLayer()
    comp_no_inj._model = model
    comp_no_inj._tokenizer = tokenizer
    comp_no_inj.set_graph(graph_no_inj)
    # No injector — answer() will retrieve but not inject

    no_inj_results = run_condition(
        "POPULATED_NO_INJECTION", model, tokenizer, injector, comp_no_inj,
        bge, N_RUNS, use_graph=True, use_injection=False
    )
    no_inj_m = print_condition("POPULATED_NO_INJECTION", no_inj_results)

    # ── Condition 3: POPULATED, WITH INJECTION ───────────────────────────────
    logger.info("\n=== Condition 3: POPULATED WITH INJECTION ===")
    from derivation.understanding_builder import UnderstandingGraph
    graph_with_inj = UnderstandingGraph(embedding_model=bge)
    populate_graph(graph_with_inj, bge)

    comp_with_inj = CompositionLayer()
    comp_with_inj._model = model
    comp_with_inj._tokenizer = tokenizer
    comp_with_inj.set_graph(graph_with_inj)
    comp_with_inj.set_injector(injector)
    # Wire local bge-m3 directly so injector doesn't fetch from HF hub
    injector._embedding_model = bge
    injector._embedding_model_loaded = True
    # Both graph and injector — full retrieve+inject pipeline

    with_inj_results = run_condition(
        "POPULATED_WITH_INJECTION", model, tokenizer, injector, comp_with_inj,
        bge, N_RUNS, use_graph=True, use_injection=True
    )
    with_inj_m = print_condition("POPULATED_WITH_INJECTION", with_inj_results)

    # ── Condition 4: RANDOM VECTOR (ablation control) ─────────────────────────
    logger.info("\n=== Condition 4: RANDOM VECTOR INJECTION (ablation control) ===")
    logger.info("  Injecting random normalized 1024-dim vector — no graph, no semantics.")
    logger.info("  Purpose: isolate whether effect is from semantic content or perturbation.")

    rand_results = run_random_vector_condition(
        "RANDOM_VECTOR", model, tokenizer, injector, bge, N_RUNS, seed=42
    )
    rand_m = print_condition("RANDOM_VECTOR", rand_results)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n=== SUMMARY ===")
    logger.info("  %-28s | hit_rate | alignment | score  | consistency", "Condition")
    logger.info("  %-28s | %8.4f | %9.4f | %6.4f | %11.4f",
                "Baseline", baseline_m["hit_rate"], baseline_m["alignment"],
                baseline_m["score"], baseline_m["consistency"])
    logger.info("  %-28s | %8.4f | %9.4f | %6.4f | %11.4f",
                "Populated no injection", no_inj_m["hit_rate"], no_inj_m["alignment"],
                no_inj_m["score"], no_inj_m["consistency"])
    logger.info("  %-28s | %8.4f | %9.4f | %6.4f | %11.4f",
                "Populated with injection", with_inj_m["hit_rate"], with_inj_m["alignment"],
                with_inj_m["score"], with_inj_m["consistency"])
    logger.info("  %-28s | %8.4f | %9.4f | %6.4f | %11.4f",
                "Random vector (ablation)", rand_m["hit_rate"], rand_m["alignment"],
                rand_m["score"], rand_m["consistency"])

    # ── Ablation verdict ─────────────────────────────────────────────────────
    threshold = 0.03  # within 3% = "similar"
    rand_vs_inj  = abs(rand_m["score"] - with_inj_m["score"])
    rand_vs_base = abs(rand_m["score"] - baseline_m["score"])

    if rand_vs_base <= threshold and rand_vs_inj > threshold:
        ablation_verdict = (
            "SCENARIO A: Random vector ~ baseline. "
            "Semantic content of projection matters. "
            "Graph-based memory hypothesis SUPPORTED."
        )
    elif rand_vs_inj <= threshold:
        ablation_verdict = (
            "SCENARIO B: Random vector ~ full injection. "
            "Effect may be from hidden-state perturbation, not semantic content. "
            "Graph-based memory claims need reframing."
        )
    else:
        ablation_verdict = (
            "AMBIGUOUS: Random vector between baseline and injection. "
            f"rand={rand_m['score']:.4f} base={baseline_m['score']:.4f} "
            f"inj={with_inj_m['score']:.4f}. "
            "Further ablation needed."
        )
    logger.info("\n  ABLATION VERDICT: %s", ablation_verdict)

    # Injection verdict
    if with_inj_m["score"] > no_inj_m["score"] > baseline_m["score"]:
        verdict = "CONTEXTUAL INJECTION WORKS — injection > graph-only > baseline"
    elif with_inj_m["score"] > baseline_m["score"]:
        verdict = "INJECTION IMPROVES over baseline; graph-only needs tuning"
    elif no_inj_m["score"] > baseline_m["score"]:
        verdict = "Graph retrieval helps; injection not yet better than graph-only"
    else:
        verdict = "injection not yet outperforming baseline — projection needs more training data"
    logger.info("  INJECTION VERDICT: %s", verdict)

    # Save results
    out = {
        "metric_version": "v42_ablation_random_vector",
        "n_runs": N_RUNS,
        "n_cases": len(TEST_CASES),
        "baseline": baseline_m,
        "populated_no_injection": no_inj_m,
        "populated_with_injection": with_inj_m,
        "random_vector_ablation": rand_m,
        "ablation_verdict": ablation_verdict,
        "verdict": verdict,
    }
    out_path = os.path.join(SCRIPT_DIR, '..', 'benchmark', 'contextual_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info("  Saved: %s", os.path.abspath(out_path))


if __name__ == '__main__':
    main()
