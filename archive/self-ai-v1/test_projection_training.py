#!/usr/bin/env python3
"""Quick test: train projection dan ukur apakah directional alignment naik."""

import os, sys, logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('test_projection_training')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dependencies', 'models'))
QWEN_PATH  = os.path.join(MODELS_DIR, 'Qwen3-0.6B')
BGE_PATH   = os.path.join(MODELS_DIR, 'bge-m3')

TRAIN_PAIRS = [
    ("kata kecuali berarti pengecualian dari kelompok", "Andi tidak lulus karena kecuali mengecualikan dia"),
    ("kata selain menunjukkan pengecualian", "harimau dikecualikan karena kata selain"),
    ("operasi pengurangan mengurangi jumlah", "35 dikurangi jumlah yang hilang"),
    ("siapa yang tidak termasuk dalam kelompok", "orang yang dikecualikan adalah jawabannya"),
    ("exception keyword reverses the included set", "the excluded one is the answer"),
    ("minus operation reduces the total count", "subtract the lost amount from total"),
    ("kata tidak termasuk berarti dikecualikan", "yang tidak termasuk adalah yang dikecualikan"),
    ("pengecualian dalam logika membalik jawaban", "jawaban adalah yang dikecualikan bukan yang termasuk"),
]

def cosine_sim(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum())

def measure_alignment(injector, bge_model, experience_text, correct_output_text):
    """Ukur cosine sim antara projected(experience_vec) dan target_hidden_state."""
    # experience vector
    emb = bge_model.encode([experience_text], normalize_embeddings=True, show_progress_bar=False)[0]
    exp_vec = torch.tensor(emb, dtype=torch.float32)

    with torch.no_grad():
        projected = injector._projection(exp_vec.unsqueeze(0)).squeeze(0)

    # target hidden state
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    inputs = tokenizer(correct_output_text, return_tensors='pt', max_length=64, truncation=True)
    captured = {}

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured['hs'] = hs[0, -1, :].detach().float()

    handle = injector.model.model.layers[injector.hook_layer_index].register_forward_hook(hook)
    with torch.no_grad():
        injector.model(**inputs)
    handle.remove()

    target = captured.get('hs')
    if target is None:
        return None

    return cosine_sim(projected.float(), target)

def main():
    logger.info("Loading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, dtype=torch.float16)
    model.eval()
    logger.info("hidden_size=%d, layers=%d", model.config.hidden_size, model.config.num_hidden_layers)

    logger.info("Loading bge-m3...")
    bge = SentenceTransformer(BGE_PATH)

    from unconscious.injector import UnconsciousInjector
    from unconscious.projection_trainer import ProjectionTrainer

    injector = UnconsciousInjector(model)

    # ── BEFORE training ──────────────────────────────────────
    logger.info("\n=== BEFORE TRAINING ===")
    before_scores = []
    for exp_text, out_text in TRAIN_PAIRS[:4]:
        score = measure_alignment(injector, bge, exp_text, out_text)
        if score is not None:
            before_scores.append(score)
            logger.info("  alignment=%.4f  exp='%s'", score, exp_text[:40])
    before_mean = sum(before_scores) / len(before_scores) if before_scores else 0
    logger.info("Before mean alignment: %.4f", before_mean)

    # ── TRAINING ─────────────────────────────────────────────
    logger.info("\n=== TRAINING (50 epochs) ===")
    trainer = ProjectionTrainer(model, tokenizer, embedding_model=bge, hook_layer_index=14)
    metrics = trainer.train(TRAIN_PAIRS, epochs=50)
    logger.info("Training done: %s", metrics)

    # Load weights into injector
    trainer.load_into_injector(injector)

    # ── AFTER training ───────────────────────────────────────
    logger.info("\n=== AFTER TRAINING ===")
    after_scores = []
    for exp_text, out_text in TRAIN_PAIRS[:4]:
        score = measure_alignment(injector, bge, exp_text, out_text)
        if score is not None:
            after_scores.append(score)
            logger.info("  alignment=%.4f  exp='%s'", score, exp_text[:40])
    after_mean = sum(after_scores) / len(after_scores) if after_scores else 0
    logger.info("After mean alignment: %.4f", after_mean)

    # ── RESULT ───────────────────────────────────────────────
    delta = after_mean - before_mean
    logger.info("\n=== RESULT ===")
    logger.info("  Before: %.4f", before_mean)
    logger.info("  After:  %.4f", after_mean)
    logger.info("  Delta:  %+.4f", delta)
    if after_mean > before_mean and after_mean > 0:
        logger.info("  ✓ PROJECTION TRAINING WORKS — alignment improved and positive")
    elif after_mean > before_mean:
        logger.info("  ~ IMPROVED but still negative — need more training pairs or epochs")
    else:
        logger.info("  ✗ NO IMPROVEMENT — check training logic")

if __name__ == '__main__':
    main()
