#!/usr/bin/env python3
# @WHO:   self-ai/tests/benchmark_steering_quality.py
# @WHAT:  Measure semantic quality of activation steering — not just "does it change?" but "does it change meaningfully?"
# @PART:  self-ai/tests
# @ENTRY: python tests/benchmark_steering_quality.py

"""Benchmark: How well does UnconsciousInjector steer Qwen3 semantically?

Unit tests (test_unconscious_injection.py) prove that injection modifies
hidden states and output tokens. But that only answers "does it change?"
— not "does it change in the RIGHT direction?"

This benchmark measures four dimensions of steering QUALITY:

1. Semantic Directional Alignment
   - Cosine similarity between (output_with_injection - output_baseline)
     and the experience vector that was injected.
   - Positive alignment = injection pushes output toward the experience.
   - This is the core metric: is the steering semantically meaningful?

2. Injection Strength Sweep
   - Run injection at strengths: 0.01, 0.05, 0.1, 0.2, 0.5
   - For each strength, measure:
     * L2 norm of hidden-state delta (how much activations change)
     * Semantic shift of output (cosine distance between output embeddings
       with and without injection)
   - Find the "sweet spot": enough shift to matter, not so much that
     output degrades into gibberish.

3. Layer Sweep
   - Inject at layers 7, 14, 21 (early / mid / late)
   - For each layer, measure semantic shift and output coherence.
   - Determine which layer produces the most effective steering.

4. Consistency Across Prompts
   - Inject the SAME experience into 5 different prompts.
   - Measure variance of the steering effect.
   - Low variance = reliable injection. High variance = unreliable.

All "semantic shift" measurements use bge-m3 to encode output text,
then compute cosine similarity/distance between embeddings.

Usage:
    cd self-ai
    python tests/benchmark_steering_quality.py

    # Or from src directory (like the existing tests):
    cd self-ai/src
    python ../tests/benchmark_steering_quality.py

Requirements:
    - Qwen3-0.6B at self-ai/dependencies/models/Qwen3-0.6B (or HF cache)
    - bge-m3 at self-ai/dependencies/models/bge-m3 (or HF cache)
    - torch, transformers, sentence_transformers
"""

import os
import sys
import logging
import time
from typing import Optional, Tuple, List, Dict

# ─── PATH SETUP ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger('benchmark_steering_quality')


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING (reuse patterns from test_unconscious_injection.py)
# ═══════════════════════════════════════════════════════════

def find_model_path(hint_path: str, hf_name: str) -> Optional[str]:
    """Find a model directory — try local path first, then HF cache."""
    abs_hint = os.path.abspath(hint_path)
    if os.path.isdir(abs_hint) and os.path.isfile(os.path.join(abs_hint, 'config.json')):
        logger.info("Found model at local path: %s", abs_hint)
        return abs_hint

    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    dir_name = f"models--{hf_name.replace('/', '--')}"
    cache_path = os.path.join(cache_dir, dir_name, 'snapshots')
    if os.path.isdir(cache_path):
        snapshots = sorted(os.listdir(cache_path))
        if snapshots:
            found = os.path.join(cache_path, snapshots[-1])
            if os.path.isfile(os.path.join(found, 'config.json')):
                logger.info("Found model in HF cache: %s", found)
                return found

    logger.info("Model not found locally — will try HF hub: %s", hf_name)
    return hf_name


def load_qwen3() -> Tuple:
    """Load Qwen3-0.6B model and tokenizer with float16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = os.path.join(SRC_DIR, '..', 'dependencies', 'models', 'Qwen3-0.6B')
    model_id = find_model_path(local_path, 'Qwen/Qwen3-0.6B')

    logger.info("Loading Qwen3-0.6B from: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    logger.info("Qwen3 loaded: %d layers, hidden_size=%d, dtype=%s, device=%s",
                num_layers, hidden_size, next(model.parameters()).dtype,
                next(model.parameters()).device)

    return model, tokenizer


def load_bge_m3():
    """Load bge-m3 embedding model."""
    from sentence_transformers import SentenceTransformer

    local_path = os.path.join(SRC_DIR, '..', 'dependencies', 'models', 'bge-m3')
    model_id = find_model_path(local_path, 'BAAI/bge-m3')

    logger.info("Loading bge-m3 from: %s", model_id)
    model = SentenceTransformer(model_id)
    dim = model.get_sentence_embedding_dimension()
    logger.info("bge-m3 loaded: embedding_dim=%d", dim)

    return model


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def make_dummy_node(embedding_model, text: str, node_id: str = 'bench-node-1'):
    """Create a dummy UnderstandingNode with a real bge-m3 embedding."""
    from derivation.understanding_builder import UnderstandingNode, Transformation
    import numpy as np

    emb = embedding_model.encode(
        [text], show_progress_bar=False, normalize_embeddings=True
    )[0]
    emb_list = emb.tolist()

    node = UnderstandingNode(
        id=node_id,
        name=f"Benchmark experience: {text[:50]}",
        concept=text,
        abstraction=f"Benchmark understanding about: {text}",
        schemas=[{"input": text, "output": "steered"}],
        transformation=Transformation(
            trigger="benchmark_trigger",
            action="steer",
            result="semantically shifted output",
        ),
        conditions=[text],
        condition_embedding=emb_list,
        source='benchmark',
        confidence=0.9,
    )
    return node


class HiddenStateCapture:
    """Capture hidden states at a specific layer for comparison."""

    def __init__(self, model, layer_index: int = 14):
        self.model = model
        self.layer_index = layer_index
        self.captured = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.captured = output[0].detach().cpu().clone()
        else:
            self.captured = output.detach().cpu().clone()

    def __enter__(self):
        layer = self.model.model.layers[self.layer_index]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        return False

    def get_captured(self):
        return self.captured


def generate_with_injection(qwen_model, tokenizer, prompt, injector, nodes,
                            layer_index=14, max_new_tokens=50):
    """Generate output with optional injection, returning (text, hidden_state)."""
    import torch

    inputs = tokenizer(prompt, return_tensors='pt')
    device = next(qwen_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with HiddenStateCapture(qwen_model, layer_index=layer_index) as capture:
            if nodes is not None:
                with injector.active(nodes):
                    output = qwen_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic for comparison
                        temperature=1.0,
                    )
            else:
                output = qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    hidden = capture.get_captured()
    return text, hidden


def semantic_cosine_sim(bge_model, text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using bge-m3."""
    import numpy as np
    if not text_a.strip() or not text_b.strip():
        return 0.0
    embeddings = bge_model.encode(
        [text_a, text_b], show_progress_bar=False, normalize_embeddings=True
    )
    return float(np.dot(embeddings[0], embeddings[1]))


def semantic_shift(bge_model, text_baseline: str, text_injected: str) -> float:
    """Semantic shift = 1 - cosine_similarity (0 = same, 1 = orthogonal, 2 = opposite)."""
    return 1.0 - semantic_cosine_sim(bge_model, text_baseline, text_injected)


def hidden_state_l2_delta(hidden_baseline, hidden_injected) -> float:
    """L2 norm of the difference at the last token position."""
    import numpy as np
    if hidden_baseline is None or hidden_injected is None:
        return 0.0
    baseline_last = hidden_baseline[0, -1, :].float().numpy()
    injected_last = hidden_injected[0, -1, :].float().numpy()
    return float(np.linalg.norm(injected_last - baseline_last))


def is_output_coherent(text: str) -> bool:
    """Quick heuristic: is the output still coherent (not garbage)?

    Checks for basic signs of degeneration:
    - Not empty
    - Not excessively repeating characters
    - Contains at least some alphanumeric characters
    """
    if not text or not text.strip():
        return False
    # Check for excessive repetition of single characters
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    # If more than 60% of characters are the same, likely garbage
    from collections import Counter
    char_counts = Counter(stripped.lower())
    if char_counts:
        most_common_ratio = char_counts.most_common(1)[0][1] / len(stripped)
        if most_common_ratio > 0.6:
            return False
    return True


# ═══════════════════════════════════════════════════════════
#  BENCHMARK 1: Semantic Directional Alignment
# ═══════════════════════════════════════════════════════════

def benchmark_directional_alignment(qwen_model, tokenizer, bge_model,
                                    experience_texts: List[str],
                                    test_prompts: List[str]) -> List[Dict]:
    """Measure cosine similarity between (output_shift) and (experience_vector).

    The key question: when injection pushes the output away from baseline,
    does it push it TOWARD the experience that was injected?

    Method:
      1. Generate baseline output (no injection)
      2. Generate injected output (with experience nodes)
      3. Encode both outputs with bge-m3
      4. Compute delta = embedding_injected - embedding_baseline
      5. Compute cosine similarity between delta and experience_vector
      6. Positive = alignment (injection pushes toward experience)
         Negative = misalignment (injection pushes away)
         Near-zero = no directional effect
    """
    import torch
    import numpy as np
    from unconscious.injector import UnconsciousInjector

    logger.info("=" * 70)
    logger.info("BENCHMARK 1: Semantic Directional Alignment")
    logger.info("=" * 70)

    results = []

    for exp_text in experience_texts:
        # Create the experience node
        node = make_dummy_node(bge_model, text=exp_text, node_id=f'dir-{exp_text[:20]}')
        nodes = [(node, 0.95)]

        # Get the raw experience vector (1024-dim, normalized) for comparison
        raw_exp_vec = np.array(node.condition_embedding, dtype=np.float32)
        raw_exp_vec = raw_exp_vec / (np.linalg.norm(raw_exp_vec) + 1e-10)

        injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.1)

        for prompt in test_prompts:
            # Generate baseline
            text_baseline, hidden_baseline = generate_with_injection(
                qwen_model, tokenizer, prompt, injector, nodes=None
            )

            # Generate with injection
            text_injected, hidden_injected = generate_with_injection(
                qwen_model, tokenizer, prompt, injector, nodes=nodes
            )

            # Encode outputs with bge-m3
            emb_baseline = bge_model.encode(
                [text_baseline], show_progress_bar=False, normalize_embeddings=True
            )[0]
            emb_injected = bge_model.encode(
                [text_injected], show_progress_bar=False, normalize_embeddings=True
            )[0]

            # Delta vector in embedding space
            delta = emb_injected - emb_baseline
            delta_norm = np.linalg.norm(delta)

            # Cosine similarity between delta and experience vector
            if delta_norm > 1e-8:
                alignment = float(np.dot(delta, raw_exp_vec))
            else:
                alignment = 0.0

            # Hidden state L2 delta
            hs_delta = hidden_state_l2_delta(hidden_baseline, hidden_injected)

            result = {
                'experience': exp_text[:40],
                'prompt': prompt[:40],
                'alignment': alignment,
                'delta_norm': delta_norm,
                'hs_l2_delta': hs_delta,
                'text_baseline': text_baseline[:80],
                'text_injected': text_injected[:80],
            }
            results.append(result)

            logger.info("  exp='%s' prompt='%s'", exp_text[:30], prompt[:30])
            logger.info("    alignment=%.4f, delta_norm=%.6f, hs_delta=%.4f",
                        alignment, delta_norm, hs_delta)

    # Summary
    alignments = [r['alignment'] for r in results]
    mean_align = np.mean(alignments)
    std_align = np.std(alignments)

    logger.info("")
    logger.info("── Directional Alignment Summary ──")
    logger.info("  Mean alignment: %.4f (positive = toward experience)", mean_align)
    logger.info("  Std alignment:  %.4f", std_align)
    logger.info("  Positive rate:  %.1f%% (%d/%d)",
                sum(1 for a in alignments if a > 0) / len(alignments) * 100,
                sum(1 for a in alignments if a > 0), len(alignments))

    return results


# ═══════════════════════════════════════════════════════════
#  BENCHMARK 2: Injection Strength Sweep
# ═══════════════════════════════════════════════════════════

def benchmark_strength_sweep(qwen_model, tokenizer, bge_model,
                             prompt: str,
                             experience_text: str,
                             strengths: List[float] = None) -> List[Dict]:
    """Sweep injection strength and measure activation change vs semantic shift.

    We need to find the sweet spot: enough strength to steer the output
    semantically, but not so much that the output degenerates.

    For each strength, we measure:
      - L2 norm of hidden-state delta (activation change)
      - Semantic shift of output (1 - cosine_sim between output embeddings)
      - Whether output is still coherent
    """
    import torch
    import numpy as np
    from unconscious.injector import UnconsciousInjector

    if strengths is None:
        strengths = [0.01, 0.05, 0.1, 0.2, 0.5]

    logger.info("=" * 70)
    logger.info("BENCHMARK 2: Injection Strength Sweep")
    logger.info("=" * 70)
    logger.info("  Prompt: '%s'", prompt[:60])
    logger.info("  Experience: '%s'", experience_text[:60])
    logger.info("  Strengths: %s", strengths)

    # Create the experience node
    node = make_dummy_node(bge_model, text=experience_text, node_id='strength-sweep')
    nodes = [(node, 0.95)]

    # Generate baseline (no injection)
    injector_baseline = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.0)
    text_baseline, hidden_baseline = generate_with_injection(
        qwen_model, tokenizer, prompt, injector_baseline, nodes=None
    )
    emb_baseline = bge_model.encode(
        [text_baseline], show_progress_bar=False, normalize_embeddings=True
    )[0]

    logger.info("  Baseline output: '%s'", text_baseline[:80])

    results = []

    for strength in strengths:
        injector = UnconsciousInjector(
            qwen_model, enabled=True, injection_strength=strength
        )

        text_injected, hidden_injected = generate_with_injection(
            qwen_model, tokenizer, prompt, injector, nodes=nodes
        )

        # Semantic shift
        emb_injected = bge_model.encode(
            [text_injected], show_progress_bar=False, normalize_embeddings=True
        )[0]
        sem_shift = 1.0 - float(np.dot(emb_baseline, emb_injected))

        # Hidden state L2 delta
        hs_delta = hidden_state_l2_delta(hidden_baseline, hidden_injected)

        # Coherence check
        coherent = is_output_coherent(text_injected)

        result = {
            'strength': strength,
            'hs_l2_delta': hs_delta,
            'semantic_shift': sem_shift,
            'coherent': coherent,
            'text': text_injected[:80],
        }
        results.append(result)

        logger.info("  strength=%.3f: hs_delta=%.4f, sem_shift=%.4f, coherent=%s",
                     strength, hs_delta, sem_shift, coherent)
        logger.info("    output: '%s'", text_injected[:80])

    # Find optimal strength: highest semantic shift while still coherent
    coherent_results = [r for r in results if r['coherent']]
    if coherent_results:
        optimal = max(coherent_results, key=lambda r: r['semantic_shift'])
        logger.info("")
        logger.info("── Strength Sweep Summary ──")
        logger.info("  Optimal strength (max semantic shift, still coherent): %.3f",
                     optimal['strength'])
        logger.info("    semantic_shift=%.4f, hs_delta=%.4f",
                     optimal['semantic_shift'], optimal['hs_l2_delta'])

        # Find breaking point: first strength where output becomes incoherent
        breaking = None
        for r in results:
            if not r['coherent']:
                breaking = r['strength']
                break
        if breaking:
            logger.info("  Breaking point (first incoherent): strength=%.3f", breaking)
        else:
            logger.info("  No breaking point found (all strengths produce coherent output)")
    else:
        logger.warning("  All strengths produced incoherent output!")

    return results


# ═══════════════════════════════════════════════════════════
#  BENCHMARK 3: Layer Sweep
# ═══════════════════════════════════════════════════════════

def benchmark_layer_sweep(qwen_model, tokenizer, bge_model,
                          prompt: str,
                          experience_text: str,
                          layers: List[int] = None,
                          strength: float = 0.1) -> List[Dict]:
    """Test injection at different transformer layers.

    Early layers (7): injection affects more downstream processing
    Mid layers (14): balanced — default in UnconsciousInjector
    Late layers (21): more targeted, less disruption

    For each layer, we measure semantic shift and output coherence.
    """
    import torch
    import numpy as np
    from unconscious.injector import UnconsciousInjector

    if layers is None:
        layers = [7, 14, 21]

    logger.info("=" * 70)
    logger.info("BENCHMARK 3: Layer Sweep")
    logger.info("=" * 70)
    logger.info("  Prompt: '%s'", prompt[:60])
    logger.info("  Experience: '%s'", experience_text[:60])
    logger.info("  Layers to test: %s", layers)

    # Create the experience node
    node = make_dummy_node(bge_model, text=experience_text, node_id='layer-sweep')
    nodes = [(node, 0.95)]

    # Generate baseline (no injection)
    injector_baseline = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.0)
    text_baseline, hidden_baseline = generate_with_injection(
        qwen_model, tokenizer, prompt, injector_baseline, nodes=None
    )
    emb_baseline = bge_model.encode(
        [text_baseline], show_progress_bar=False, normalize_embeddings=True
    )[0]

    logger.info("  Baseline output: '%s'", text_baseline[:80])

    results = []

    for layer_idx in layers:
        injector = UnconsciousInjector(
            qwen_model, enabled=True,
            injection_strength=strength,
            hook_layer_index=layer_idx,
        )

        text_injected, hidden_injected = generate_with_injection(
            qwen_model, tokenizer, prompt, injector, nodes=nodes,
            layer_index=layer_idx,
        )

        # Semantic shift
        emb_injected = bge_model.encode(
            [text_injected], show_progress_bar=False, normalize_embeddings=True
        )[0]
        sem_shift = 1.0 - float(np.dot(emb_baseline, emb_injected))

        # Hidden state L2 delta at the injection layer
        hs_delta = hidden_state_l2_delta(hidden_baseline, hidden_injected)

        # Coherence check
        coherent = is_output_coherent(text_injected)

        result = {
            'layer': layer_idx,
            'position': 'early' if layer_idx < 10 else ('mid' if layer_idx < 20 else 'late'),
            'hs_l2_delta': hs_delta,
            'semantic_shift': sem_shift,
            'coherent': coherent,
            'text': text_injected[:80],
        }
        results.append(result)

        logger.info("  layer=%2d (%s): hs_delta=%.4f, sem_shift=%.4f, coherent=%s",
                     layer_idx, result['position'], hs_delta, sem_shift, coherent)
        logger.info("    output: '%s'", text_injected[:80])

    # Find best layer
    coherent_results = [r for r in results if r['coherent']]
    if coherent_results:
        best = max(coherent_results, key=lambda r: r['semantic_shift'])
        logger.info("")
        logger.info("── Layer Sweep Summary ──")
        logger.info("  Best layer (max semantic shift, still coherent): %d (%s)",
                     best['layer'], best['position'])
        logger.info("    semantic_shift=%.4f, hs_delta=%.4f",
                     best['semantic_shift'], best['hs_l2_delta'])
    else:
        logger.warning("  All layers produced incoherent output!")

    return results


# ═══════════════════════════════════════════════════════════
#  BENCHMARK 4: Consistency Across Prompts
# ═══════════════════════════════════════════════════════════

def benchmark_consistency(qwen_model, tokenizer, bge_model,
                          experience_text: str,
                          prompts: List[str],
                          strength: float = 0.1) -> Dict:
    """Inject the SAME experience into different prompts, measure variance.

    If injection is reliable, the steering effect should be consistent
    across prompts. High variance means the injection is not dependable —
    it works for some prompts but not others.

    We measure:
      - For each prompt: semantic shift (with vs without injection)
      - Variance of semantic shifts across prompts
      - Whether injection always changes output (consistency rate)
    """
    import torch
    import numpy as np
    from unconscious.injector import UnconsciousInjector

    logger.info("=" * 70)
    logger.info("BENCHMARK 4: Consistency Across Prompts")
    logger.info("=" * 70)
    logger.info("  Experience: '%s'", experience_text[:60])
    logger.info("  Prompts: %d", len(prompts))

    # Create the experience node
    node = make_dummy_node(bge_model, text=experience_text, node_id='consistency')
    nodes = [(node, 0.95)]

    injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=strength)

    shifts = []
    per_prompt_results = []

    for i, prompt in enumerate(prompts):
        # Baseline
        text_baseline, _ = generate_with_injection(
            qwen_model, tokenizer, prompt, injector, nodes=None
        )

        # With injection
        text_injected, _ = generate_with_injection(
            qwen_model, tokenizer, prompt, injector, nodes=nodes
        )

        # Semantic shift
        shift = semantic_shift(bge_model, text_baseline, text_injected)

        # Did output actually change?
        output_changed = text_baseline != text_injected

        shifts.append(shift)
        per_prompt_results.append({
            'prompt': prompt[:50],
            'semantic_shift': shift,
            'output_changed': output_changed,
            'text_baseline': text_baseline[:60],
            'text_injected': text_injected[:60],
        })

        logger.info("  prompt %d: '%s'", i + 1, prompt[:40])
        logger.info("    shift=%.4f, output_changed=%s", shift, output_changed)

    # Compute consistency metrics
    shifts_arr = np.array(shifts)
    variance = float(np.var(shifts_arr))
    mean_shift = float(np.mean(shifts_arr))
    std_shift = float(np.std(shifts_arr))
    consistency_rate = sum(1 for r in per_prompt_results if r['output_changed']) / len(prompts)
    cv = std_shift / (mean_shift + 1e-10)  # Coefficient of variation

    logger.info("")
    logger.info("── Consistency Summary ──")
    logger.info("  Mean semantic shift: %.4f", mean_shift)
    logger.info("  Std semantic shift:  %.4f", std_shift)
    logger.info("  Variance:            %.6f", variance)
    logger.info("  Coeff of variation:  %.4f (lower = more consistent)", cv)
    logger.info("  Output change rate:  %.1f%% (%d/%d)",
                consistency_rate * 100,
                sum(1 for r in per_prompt_results if r['output_changed']),
                len(prompts))

    if cv < 0.5:
        logger.info("  Verdict: CONSISTENT (CV < 0.5)")
    elif cv < 1.0:
        logger.info("  Verdict: MODERATELY CONSISTENT (0.5 <= CV < 1.0)")
    else:
        logger.info("  Verdict: INCONSISTENT (CV >= 1.0)")

    return {
        'mean_shift': mean_shift,
        'std_shift': std_shift,
        'variance': variance,
        'cv': cv,
        'consistency_rate': consistency_rate,
        'per_prompt': per_prompt_results,
    }


# ═══════════════════════════════════════════════════════════
#  TABLE FORMATTING
# ═══════════════════════════════════════════════════════════

def print_separator(char='-', width=90):
    print(char * width)


def print_table(headers: List[str], rows: List[List[str]], col_widths: List[int] = None):
    """Print a formatted table to stdout."""
    if col_widths is None:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]

    # Header
    header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print_separator('-', len(header_line))

    # Rows
    for row in rows:
        row_line = ' | '.join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(row_line)


# ═══════════════════════════════════════════════════════════
#  MAIN BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════

def main():
    """Run all benchmarks and print human-readable tables."""
    import numpy as np

    start_time = time.time()

    print()
    print_separator('=')
    print("  SELF-AI Steering Quality Benchmark")
    print("  Measures: directional alignment, strength sweep, layer sweep, consistency")
    print_separator('=')
    print()

    # ─── Load models ───
    logger.info("Loading models...")
    qwen_model, tokenizer = load_qwen3()
    bge_model = load_bge_m3()
    logger.info("Models loaded. Starting benchmarks...\n")

    # ─── Test data ───
    experience_texts = [
        "kata pengecualian kecuali selain tidak membalik jawaban",
        "operasi pengurangan kehilangan terjual berkurang hilang",
        "pertanyaan negatif bukan tidak jangan kebalikan",
    ]

    test_prompts = [
        "Semua siswa lulus kecuali Andi, siapa yang tidak lulus?",
        "Toko kehilangan 35 roti, berapa sisa?",
        "Semua hewan jinak selain harimau, mana yang bukan jinak?",
        "Jawaban yang benar bukan A, melainkan B. Mana yang benar?",
        "Tidak ada yang gagal kecuali Budi, siapa yang gagal?",
    ]

    # ─── BENCHMARK 1: Directional Alignment ───
    print()
    print_separator('=')
    print("  BENCHMARK 1: Semantic Directional Alignment")
    print("  Cosine similarity between output-shift and experience vector")
    print_separator('=')

    dir_results = benchmark_directional_alignment(
        qwen_model, tokenizer, bge_model,
        experience_texts=experience_texts[:2],  # Use 2 experiences
        test_prompts=test_prompts[:3],          # Use 3 prompts
    )

    # Print table
    dir_rows = []
    for r in dir_results:
        dir_rows.append([
            r['experience'][:25],
            r['prompt'][:25],
            f"{r['alignment']:.4f}",
            f"{r['delta_norm']:.6f}",
            f"{r['hs_l2_delta']:.4f}",
            "ALIGNED" if r['alignment'] > 0 else "MISALIGNED",
        ])
    print()
    print_table(
        headers=['Experience', 'Prompt', 'Alignment', 'Delta Norm', 'HS L2 Δ', 'Verdict'],
        rows=dir_rows,
        col_widths=[26, 26, 10, 12, 10, 12],
    )

    # Aggregate
    alignments = [r['alignment'] for r in dir_results]
    mean_align = np.mean(alignments)
    print()
    print(f"  Mean alignment: {mean_align:.4f} ({'POSITIVE → injection pushes toward experience' if mean_align > 0 else 'NEGATIVE → injection pushes away'})")

    # ─── BENCHMARK 2: Strength Sweep ───
    print()
    print_separator('=')
    print("  BENCHMARK 2: Injection Strength Sweep")
    print("  Finding optimal strength before output degrades")
    print_separator('=')

    strength_results = benchmark_strength_sweep(
        qwen_model, tokenizer, bge_model,
        prompt=test_prompts[0],
        experience_text=experience_texts[0],
        strengths=[0.01, 0.05, 0.1, 0.2, 0.5],
    )

    # Print table
    str_rows = []
    for r in strength_results:
        str_rows.append([
            f"{r['strength']:.3f}",
            f"{r['hs_l2_delta']:.4f}",
            f"{r['semantic_shift']:.4f}",
            "YES" if r['coherent'] else "NO",
            r['text'][:30],
        ])
    print()
    print_table(
        headers=['Strength', 'HS L2 Δ', 'Sem Shift', 'Coherent', 'Output Preview'],
        rows=str_rows,
        col_widths=[9, 10, 10, 9, 32],
    )

    # ─── BENCHMARK 3: Layer Sweep ───
    print()
    print_separator('=')
    print("  BENCHMARK 3: Layer Sweep (early / mid / late)")
    print("  Which transformer layer is most effective for steering?")
    print_separator('=')

    layer_results = benchmark_layer_sweep(
        qwen_model, tokenizer, bge_model,
        prompt=test_prompts[0],
        experience_text=experience_texts[0],
        layers=[7, 14, 21],
        strength=0.1,
    )

    # Print table
    lay_rows = []
    for r in layer_results:
        lay_rows.append([
            f"{r['layer']}",
            r['position'],
            f"{r['hs_l2_delta']:.4f}",
            f"{r['semantic_shift']:.4f}",
            "YES" if r['coherent'] else "NO",
            r['text'][:30],
        ])
    print()
    print_table(
        headers=['Layer', 'Position', 'HS L2 Δ', 'Sem Shift', 'Coherent', 'Output Preview'],
        rows=lay_rows,
        col_widths=[6, 8, 10, 10, 9, 32],
    )

    # ─── BENCHMARK 4: Consistency ───
    print()
    print_separator('=')
    print("  BENCHMARK 4: Consistency Across Prompts")
    print("  Same experience, different prompts — how reliable is the steering?")
    print_separator('=')

    consistency_results = benchmark_consistency(
        qwen_model, tokenizer, bge_model,
        experience_text=experience_texts[0],
        prompts=test_prompts,
        strength=0.1,
    )

    # Print per-prompt table
    con_rows = []
    for r in consistency_results['per_prompt']:
        con_rows.append([
            r['prompt'][:30],
            f"{r['semantic_shift']:.4f}",
            "CHANGED" if r['output_changed'] else "SAME",
        ])
    print()
    print_table(
        headers=['Prompt', 'Sem Shift', 'Output'],
        rows=con_rows,
        col_widths=[32, 10, 8],
    )

    # Print aggregate
    print()
    print(f"  Mean shift:       {consistency_results['mean_shift']:.4f}")
    print(f"  Std shift:        {consistency_results['std_shift']:.4f}")
    print(f"  Variance:         {consistency_results['variance']:.6f}")
    print(f"  CV:               {consistency_results['cv']:.4f} (lower = more consistent)")
    print(f"  Change rate:      {consistency_results['consistency_rate']:.0%}")

    # ─── FINAL SUMMARY ───
    elapsed = time.time() - start_time

    print()
    print_separator('=')
    print("  BENCHMARK SUMMARY")
    print_separator('=')
    print()
    print(f"  1. Directional Alignment:   mean={mean_align:.4f} ({'POSITIVE' if mean_align > 0 else 'NEGATIVE'})")
    print(f"  2. Strength Sweep:          optimal={max((r for r in strength_results if r['coherent']), key=lambda r: r['semantic_shift'])['strength']:.3f}" if any(r['coherent'] for r in strength_results) else "  2. Strength Sweep:          NO coherent output found")
    print(f"  3. Layer Sweep:             best_layer={max((r for r in layer_results if r['coherent']), key=lambda r: r['semantic_shift'])['layer']}" if any(r['coherent'] for r in layer_results) else "  3. Layer Sweep:             NO coherent output found")
    print(f"  4. Consistency:             CV={consistency_results['cv']:.4f}, change_rate={consistency_results['consistency_rate']:.0%}")
    print()
    print(f"  Total time: {elapsed:.1f}s")
    print()

    # ─── TECHNICAL ASSUMPTIONS ───
    print_separator('=')
    print("  TECHNICAL ASSUMPTIONS (validate when running on machine with models)")
    print_separator('=')
    print("""
  1. Qwen3-0.6B has exactly 28 transformer layers (model.model.layers)
  2. Qwen3 hidden_size = 896 (read from model.config.hidden_size)
  3. bge-m3 embedding dim = 1024
  4. Qwen3 loaded with torch.float16, device_map='auto'
  5. Deterministic generation (do_sample=False) for reproducible comparisons
  6. Projection 1024→896 uses identity-like initialization (NOT trained)
  7. Injection is additive at last token position only
  8. Semantic shift measured via bge-m3 cosine distance on generated text
  9. Coherence check is heuristic (repetition ratio, min length) — not a
     full language model perplexity check
  10. Directional alignment assumes embedding delta direction correlates
      with experience vector direction — this is an assumption that needs
      empirical validation
""")

    return {
        'directional_alignment': dir_results,
        'strength_sweep': strength_results,
        'layer_sweep': layer_results,
        'consistency': consistency_results,
    }


if __name__ == '__main__':
    main()
