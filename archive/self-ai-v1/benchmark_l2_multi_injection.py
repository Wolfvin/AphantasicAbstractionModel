#!/usr/bin/env python3
"""Benchmark L2 multi-experience injection vs single-node vs baseline.

Metrics (designed for long-term reuse):
  1. keyword_hit_rate  — does the correct answer keyword appear in output? (0.0–1.0)
  2. answer_alignment  — cosine sim of output embedding to *correct answer text* embedding
                         (more specific than cosine-to-theme-label)
  3. semantic_score    — weighted combo: 0.6 * keyword_hit_rate + 0.4 * answer_alignment

Why these metrics:
  - keyword_hit_rate is binary/exact — no embedding noise, easy to interpret
  - answer_alignment uses the full correct answer, not a 3-word label
  - Together they catch both surface correctness and semantic direction
  - Both are reusable: add new TEST_CASES without changing metric logic
"""

import os, sys, logging, json, statistics, uuid, re
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('benchmark_l2')
logger.setLevel(logging.INFO)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dependencies', 'models'))
QWEN_PATH = os.path.join(MODELS_DIR, 'Qwen3-0.6B')
BGE_PATH  = os.path.join(MODELS_DIR, 'bge-m3')

# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry needs:
#   question          — prompt sent to model
#   correct_answer    — full correct answer text (used for answer_alignment)
#   correct_keywords  — list of strings that should appear in output (OR logic)
#   experiences       — related experience texts to inject
TEST_CASES = [
    {
        "question": "Sebutkan satu hewan yang tidak termasuk dalam kelompok: kucing, anjing, ikan, harimau.",
        "correct_answer": "harimau tidak termasuk dalam kelompok hewan peliharaan",
        "correct_keywords": ["harimau"],
        "experiences": [
            "kata kecuali berarti pengecualian dari suatu kelompok",
            "kata selain menunjukkan elemen yang tidak termasuk",
            "dalam logika pengecualian jawaban adalah yang dikecualikan",
        ],
    },
    {
        "question": "Berapa hasil dari 50 dikurangi jumlah yang hilang jika ada 15 yang hilang?",
        "correct_answer": "35, karena 50 dikurangi 15 sama dengan 35",
        "correct_keywords": ["35"],
        "experiences": [
            "operasi pengurangan mengurangi jumlah awal",
            "kata dikurangi berarti menghitung selisih",
        ],
    },
    {
        "question": "Siapa yang tidak lulus jika semua lulus kecuali Budi?",
        "correct_answer": "Budi adalah yang tidak lulus",
        "correct_keywords": ["budi"],
        "experiences": [
            "kata kecuali mengidentifikasi yang tidak termasuk",
            "yang disebut setelah kecuali adalah jawabannya",
            "pengecualian dalam kalimat menunjuk satu entitas",
        ],
    },
    {
        "question": "Sebutkan bilangan yang bukan bilangan prima dari: 2, 3, 4, 5.",
        "correct_answer": "4 bukan bilangan prima karena dapat dibagi 2",
        "correct_keywords": ["4"],
        "experiences": [
            "bilangan prima hanya bisa dibagi 1 dan dirinya sendiri",
            "bilangan yang bukan prima memiliki faktor selain 1 dan dirinya",
        ],
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
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(
        out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()


def make_node(experience_text: str, confidence: float, bge):
    from derivation.understanding_builder import UnderstandingNode
    node = UnderstandingNode(
        id=str(uuid.uuid4()),
        name=experience_text[:30],
        concept=experience_text,
        abstraction="benchmark",
        confidence=confidence,
    )
    emb = bge.encode([experience_text], normalize_embeddings=True, show_progress_bar=False)[0]
    node.condition_embedding = emb.tolist()
    return node


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_condition(label, model, tokenizer, injector, node_tuples, bge, n_runs):
    """
    Run all TEST_CASES under one injection condition.
    node_tuples: list of (UnderstandingNode, score) or [] for baseline.
    Returns list of per-case result dicts.
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
            if node_tuples:
                with injector.active(node_tuples):
                    text = generate(model, tokenizer, tc["question"], **GEN_KWARGS)
            else:
                text = generate(model, tokenizer, tc["question"], **GEN_KWARGS)
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


def main():
    logger.info("Loading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, dtype=torch.float16)
    model.eval()

    logger.info("Loading bge-m3...")
    bge = SentenceTransformer(BGE_PATH)

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

    # ── Build node sets ───────────────────────────────────────────────────────
    # Use experiences from ALL test cases as universal injection context
    single_tuples = [(make_node(TEST_CASES[0]["experiences"][0], 0.8, bge), 0.8)]

    multi_tuples = [
        (make_node(exp, round(0.9 - j * 0.1, 2), bge), round(0.9 - j * 0.1, 2))
        for tc in TEST_CASES
        for j, exp in enumerate(tc["experiences"])
    ]

    # ── Run conditions ────────────────────────────────────────────────────────
    logger.info("\n=== BENCHMARK START (metric: keyword_hit + answer_alignment) ===")

    baseline_m = print_condition("BASELINE",    run_condition("BASELINE",    model, tokenizer, injector, [],             bge, N_RUNS))
    single_m   = print_condition("SINGLE-NODE", run_condition("SINGLE-NODE", model, tokenizer, injector, single_tuples, bge, N_RUNS))
    multi_m    = print_condition("MULTI-NODE",  run_condition("MULTI-NODE",  model, tokenizer, injector, multi_tuples,  bge, N_RUNS))

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n=== SUMMARY ===")
    logger.info("  %-12s | hit_rate | alignment | score  | consistency", "Condition")
    logger.info("  %-12s | %8.4f | %9.4f | %6.4f | %11.4f", "Baseline",    baseline_m["hit_rate"], baseline_m["alignment"], baseline_m["score"], baseline_m["consistency"])
    logger.info("  %-12s | %8.4f | %9.4f | %6.4f | %11.4f", "Single-node", single_m["hit_rate"],   single_m["alignment"],   single_m["score"],   single_m["consistency"])
    logger.info("  %-12s | %8.4f | %9.4f | %6.4f | %11.4f", "Multi-node",  multi_m["hit_rate"],    multi_m["alignment"],    multi_m["score"],    multi_m["consistency"])

    # Verdict
    if multi_m["score"] > single_m["score"] > baseline_m["score"]:
        verdict = "L2 WORKS — multi-node > single > baseline"
    elif multi_m["score"] > baseline_m["score"]:
        verdict = "L2 IMPROVES over baseline; single-node needs tuning"
    elif single_m["score"] > baseline_m["score"]:
        verdict = "single-node helps; L2 not yet better than single"
    else:
        verdict = "injection not yet outperforming baseline — projection needs more training data"
    logger.info("\n  VERDICT: %s", verdict)

    # Save results
    out = {
        "metric_version": "v2_keyword_hit_plus_answer_alignment",
        "n_runs": N_RUNS,
        "n_cases": len(TEST_CASES),
        "baseline":    baseline_m,
        "single_node": single_m,
        "multi_node":  multi_m,
        "verdict": verdict,
    }
    out_path = os.path.join(SCRIPT_DIR, '..', 'benchmark', 'l2_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info("  Saved: %s", os.path.abspath(out_path))


if __name__ == '__main__':
    main()
