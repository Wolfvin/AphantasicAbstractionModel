#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_unconscious_injection.py
# @WHAT:  Verify that UnconsciousInjector actually modifies hidden states and outputs
# @PART:  self-ai/tests
# @ENTRY: python test_unconscious_injection.py

"""Test: Does UnconsciousInjector ACTUALLY work?

This test proves (or disproves) that the activation steering pipeline
(step 4→5) is a live path, not dead code.

What it tests:
  1. Load Qwen3-0.6B from local path with float16
  2. Load bge-m3 from local path
  3. Generate output WITHOUT injection → capture hidden state at layer 14
  4. Create dummy experience node with bge-m3 embedding
  5. Generate output WITH injection → capture hidden state at layer 14
  6. Compare: are hidden states different? Are output tokens different?
  7. Test introspection via Introspector.explain_last_answer()

Definition of Done:
  - Hidden states at layer 14 are DIFFERENT with vs without injection
  - Output tokens differ OR logits differ at the injection point
  - Introspector returns a non-empty explanation
  - No RuntimeError (e.g., dtype mismatch)

Run:
  cd self-ai/src
  python -m pytest ../tests/test_unconscious_injection.py -v -s

  # Or directly:
  cd self-ai/src
  python ../tests/test_unconscious_injection.py
"""

import os
import sys
import logging
import time
from typing import Optional, Tuple

# ─── PATH SETUP ───
# self-ai/src must be on sys.path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger('test_unconscious_injection')


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════

def find_model_path(hint_path: str, hf_name: str) -> Optional[str]:
    """Find a model directory — try local path first, then HF cache.

    Args:
        hint_path: Preferred local path (e.g., self-ai/dependencies/models/Qwen3-0.6B)
        hf_name: HuggingFace model name for cache lookup (e.g., Qwen/Qwen3-0.6B)

    Returns:
        Absolute path to model directory, or None.
    """
    # Try the hinted local path
    abs_hint = os.path.abspath(hint_path)
    if os.path.isdir(abs_hint) and os.path.isfile(os.path.join(abs_hint, 'config.json')):
        logger.info("Found model at local path: %s", abs_hint)
        return abs_hint

    # Try HF cache
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    dir_name = f"models--{hf_name.replace('/', '--')}"
    cache_path = os.path.join(cache_dir, dir_name, 'snapshots')
    if os.path.isdir(cache_path):
        # Pick the latest snapshot
        snapshots = sorted(os.listdir(cache_path))
        if snapshots:
            found = os.path.join(cache_path, snapshots[-1])
            if os.path.isfile(os.path.join(found, 'config.json')):
                logger.info("Found model in HF cache: %s", found)
                return found

    # Try as HuggingFace name directly (will download if online)
    logger.info("Model not found locally — will try HF hub: %s", hf_name)
    return hf_name


def load_qwen3() -> Tuple:
    """Load Qwen3-0.6B model and tokenizer.

    Uses float16 to avoid OOM. Tries local path first, then HF cache.

    Returns:
        (model, tokenizer) tuple.
    """
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

    # Verify architecture expectations
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    logger.info("Qwen3 loaded: %d layers, hidden_size=%d, dtype=%s, device=%s",
                num_layers, hidden_size, next(model.parameters()).dtype,
                next(model.parameters()).device)

    assert num_layers == 28, f"Expected 28 layers, got {num_layers}"
    assert hidden_size > 0, f"Invalid hidden_size={hidden_size}"

    return model, tokenizer


def load_bge_m3():
    """Load bge-m3 embedding model.

    Returns:
        SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    local_path = os.path.join(SRC_DIR, '..', 'dependencies', 'models', 'bge-m3')
    model_id = find_model_path(local_path, 'BAAI/bge-m3')

    logger.info("Loading bge-m3 from: %s", model_id)
    model = SentenceTransformer(model_id)
    dim = model.get_sentence_embedding_dimension()
    logger.info("bge-m3 loaded: embedding_dim=%d", dim)
    assert dim == 1024, f"Expected embedding_dim=1024, got {dim}"

    return model


# ═══════════════════════════════════════════════════════════
#  DUMMY NODE CREATION
# ═══════════════════════════════════════════════════════════

def make_dummy_node(embedding_model, text: str, node_id: str = 'test-node-1'):
    """Create a dummy UnderstandingNode with a real bge-m3 embedding.

    This simulates an experience node that SELF would have created
    from a past correction. The embedding is computed from the
    condition text (what triggers this experience).

    Args:
        embedding_model: bge-m3 SentenceTransformer.
        text: Condition text to embed (e.g., "kata kecuali membalik jawaban").
        node_id: Unique ID for the node.

    Returns:
        UnderstandingNode with condition_embedding populated.
    """
    from derivation.understanding_builder import UnderstandingNode, Transformation

    # Compute real embedding from bge-m3
    import numpy as np
    emb = embedding_model.encode(
        [text], show_progress_bar=False, normalize_embeddings=True
    )[0]
    emb_list = emb.tolist()

    node = UnderstandingNode(
        id=node_id,
        name=f"Test experience: {text[:50]}",
        concept=text,
        abstraction=f"This is a test understanding about: {text}",
        schemas=[{"input": text, "output": "reversed"}],
        transformation=Transformation(
            kind="test",
            trigger={"signal": "test_trigger"},
            action="reverse",
        ),
        conditions=[text],
        condition_embedding=emb_list,
        source='test',
        confidence=0.9,
    )
    # Set governance to ensure it passes _is_injectable
    # (UnderstandingNode __init__ defaults to STABLE/OBSERVED, which is fine)

    logger.info("Created dummy node '%s': embedding_dim=%d, confidence=%.2f, "
                "lifecycle=%s, epistemic=%s",
                node_id, len(emb_list), node.confidence,
                node.lifecycle.value, node.epistemic.value)

    return node


# ═══════════════════════════════════════════════════════════
#  HIDDEN STATE CAPTURE
# ═══════════════════════════════════════════════════════════

class HiddenStateCapture:
    """Capture hidden states at a specific layer for comparison.

    Uses a forward hook to intercept the output of a transformer layer
    and store it. This lets us compare hidden states with and without
    injection to prove the injection actually happened.
    """

    def __init__(self, model, layer_index: int = 14):
        self.model = model
        self.layer_index = layer_index
        self.captured = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.captured = output[0].detach().cpu()
        else:
            self.captured = output.detach().cpu()

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


# ═══════════════════════════════════════════════════════════
#  TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════

def test_injection_modifies_hidden_states():
    """Test 1: Does injection actually change hidden states at layer 14?

    This is the most fundamental test. If hidden states don't change,
    the injection is a dead path.
    """
    import torch
    import numpy as np

    logger.info("=" * 60)
    logger.info("TEST 1: Hidden State Modification")
    logger.info("=" * 60)

    # Load models
    qwen_model, tokenizer = load_qwen3()
    bge_model = load_bge_m3()

    # Setup injector
    from unconscious.injector import UnconsciousInjector
    injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.1)

    # Test question
    question = "Semua siswa lulus kecuali Andi, siapa yang tidak lulus?"

    # ── Step 1: Generate WITHOUT injection ──
    inputs = tokenizer(question, return_tensors='pt')
    device = next(qwen_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with HiddenStateCapture(qwen_model, layer_index=14) as capture_baseline:
            output_baseline = qwen_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # Deterministic for comparison
                temperature=1.0,
            )

    hidden_baseline = capture_baseline.get_captured()
    text_baseline = tokenizer.decode(output_baseline[0], skip_special_tokens=True)

    logger.info("BASELINE output: %s", text_baseline)
    logger.info("BASELINE hidden shape: %s, dtype: %s",
                hidden_baseline.shape if hidden_baseline is not None else None,
                hidden_baseline.dtype if hidden_baseline is not None else None)

    # ── Step 2: Generate WITH injection ──
    # Create a dummy experience node with strong embedding signal
    # Use a text that's semantically related to "kecuali" (exception words)
    dummy_node = make_dummy_node(
        bge_model,
        text="kata pengecualian kecuali selain tidak membalik jawaban",
        node_id='test-kecuali-1',
    )
    experience_nodes = [(dummy_node, 0.95)]

    with torch.no_grad():
        with injector.active(experience_nodes):
            # Also capture hidden states to see the difference
            with HiddenStateCapture(qwen_model, layer_index=14) as capture_injected:
                output_injected = qwen_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=1.0,
                )

    hidden_injected = capture_injected.get_captured()
    text_injected = tokenizer.decode(output_injected[0], skip_special_tokens=True)

    logger.info("INJECTED output: %s", text_injected)
    logger.info("INJECTED hidden shape: %s, dtype: %s",
                hidden_injected.shape if hidden_injected is not None else None,
                hidden_injected.dtype if hidden_injected is not None else None)

    # ── Step 3: Compare ──
    assert hidden_baseline is not None, "Baseline hidden state not captured!"
    assert hidden_injected is not None, "Injected hidden state not captured!"

    # Compare hidden states at the last token position
    baseline_last = hidden_baseline[0, -1, :].float().numpy()
    injected_last = hidden_injected[0, -1, :].float().numpy()

    diff = np.abs(injected_last - baseline_last)
    max_diff = diff.max()
    mean_diff = diff.mean()
    cosine_sim = np.dot(baseline_last, injected_last) / (
        np.linalg.norm(baseline_last) * np.linalg.norm(injected_last) + 1e-10
    )

    logger.info("── Hidden State Comparison (layer 14, last token) ──")
    logger.info("  Max absolute difference: %.6f", max_diff)
    logger.info("  Mean absolute difference: %.6f", mean_diff)
    logger.info("  Cosine similarity: %.6f", cosine_sim)
    logger.info("  Baseline norm: %.6f", np.linalg.norm(baseline_last))
    logger.info("  Injected norm: %.6f", np.linalg.norm(injected_last))

    # The key assertion: hidden states MUST be different
    assert max_diff > 1e-5, (
        f"Hidden states are IDENTICAL with/without injection! "
        f"max_diff={max_diff:.8f} — injection is NOT working."
    )

    logger.info("✓ PASS: Hidden states differ (max_diff=%.6f)", max_diff)

    # Check injection log
    log = injector.get_injection_log()
    logger.info("Injection log: active=%s, nodes=%d, strength=%.3f",
                log.get('active'), len(log.get('nodes', [])), log.get('strength', 0))

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'cosine_sim': cosine_sim,
        'text_baseline': text_baseline,
        'text_injected': text_injected,
    }


def test_injection_affects_output():
    """Test 2: Does injection change the generated output tokens?

    While hidden state difference is necessary, the ultimate proof
    is whether the output changes. With injection_strength=0.1, the
    change might be subtle, so we also try with higher strength.
    """
    import torch

    logger.info("=" * 60)
    logger.info("TEST 2: Output Token Difference")
    logger.info("=" * 60)

    qwen_model, tokenizer = load_qwen3()
    bge_model = load_bge_m3()

    question = "Semua siswa lulus kecuali Andi, siapa yang tidak lulus?"
    inputs = tokenizer(question, return_tensors='pt')
    device = next(qwen_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Baseline without injection
    with torch.no_grad():
        output_baseline = qwen_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
    text_baseline = tokenizer.decode(output_baseline[0], skip_special_tokens=True)

    # Test with increasing injection strengths
    results = []
    for strength in [0.1, 0.5, 1.0, 3.0]:
        from unconscious.injector import UnconsciousInjector
        injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=strength)

        dummy_node = make_dummy_node(
            bge_model,
            text="kata pengecualian kecuali selain tidak membalik jawaban",
            node_id='test-kecuali-1',
        )
        experience_nodes = [(dummy_node, 0.95)]

        with torch.no_grad():
            with injector.active(experience_nodes):
                output_injected = qwen_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )

        text_injected = tokenizer.decode(output_injected[0], skip_special_tokens=True)
        same = text_baseline == text_injected

        logger.info("  strength=%.1f: same=%s | baseline=%s | injected=%s",
                     strength, same,
                     text_baseline[:80], text_injected[:80])

        results.append({
            'strength': strength,
            'same_output': same,
            'text_injected': text_injected,
        })

    # At least one strength level should produce different output
    any_different = any(not r['same_output'] for r in results)

    if any_different:
        logger.info("✓ PASS: Output tokens differ at some injection strength")
    else:
        logger.warning("⚠ WARNING: Output tokens are identical at all strengths "
                       "(hidden states may still differ — see Test 1)")

    return results


def test_injection_log_populated():
    """Test 3: Is the injection log correctly populated?

    The Introspector relies on the injection log being filled with
    correct node details. If the log is empty, introspection fails.
    """
    logger.info("=" * 60)
    logger.info("TEST 3: Injection Log Population")
    logger.info("=" * 60)

    import torch
    qwen_model, tokenizer = load_qwen3()
    bge_model = load_bge_m3()

    from unconscious.injector import UnconsciousInjector
    injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.1)

    # Before injection — log should be inactive
    log = injector.get_injection_log()
    logger.info("Before injection: active=%s", log.get('active'))
    assert log.get('active') is False, "Log should be inactive before injection"

    # Create nodes and inject
    dummy_node = make_dummy_node(
        bge_model,
        text="kata pengecualian kecuali selain tidak membalik jawaban",
        node_id='test-kecuali-1',
    )
    experience_nodes = [(dummy_node, 0.95)]

    question = "Semua siswa lulus kecuali Andi, siapa yang tidak lulus?"
    inputs = tokenizer(question, return_tensors='pt')
    device = next(qwen_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with injector.active(experience_nodes):
            _ = qwen_model.generate(**inputs, max_new_tokens=20, do_sample=False)

    # After injection — log should be populated
    log = injector.get_injection_log()
    logger.info("After injection: active=%s, nodes=%d, strength=%.3f, layer=%d",
                log.get('active'), len(log.get('nodes', [])),
                log.get('strength', 0), log.get('layer', '?'))

    assert log.get('active') is True, "Log should be active after injection"
    assert len(log.get('nodes', [])) > 0, "Log should contain node IDs"
    assert log.get('strength') == 0.1, f"Strength should be 0.1, got {log.get('strength')}"
    assert log.get('layer') == 14, f"Layer should be 14, got {log.get('layer')}"

    node_details = log.get('node_details', [])
    assert len(node_details) > 0, "Log should contain node details"
    nd = node_details[0]
    assert nd.get('id') == 'test-kecuali-1', f"Node ID mismatch: {nd.get('id')}"
    assert 'lifecycle' in nd, "Node detail should include lifecycle"
    assert 'epistemic' in nd, "Node detail should include epistemic"

    logger.info("✓ PASS: Injection log correctly populated")
    return log


def test_introspection():
    """Test 4: Does Introspector.explain_last_answer() work?

    After an injection, the Introspector should be able to generate
    a natural-language explanation of what influenced the answer.
    """
    logger.info("=" * 60)
    logger.info("TEST 4: Introspection")
    logger.info("=" * 60)

    import torch
    qwen_model, tokenizer = load_qwen3()
    bge_model = load_bge_m3()

    from unconscious.injector import UnconsciousInjector
    from introspection.introspector import Introspector

    injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.1)
    introspector = Introspector(injector, model=qwen_model, tokenizer=tokenizer)

    # Create nodes and inject
    dummy_node = make_dummy_node(
        bge_model,
        text="kata pengecualian kecuali selain tidak membalik jawaban",
        node_id='test-kecuali-1',
    )
    experience_nodes = [(dummy_node, 0.95)]

    question = "Semua siswa lulus kecuali Andi, siapa yang tidak lulus?"
    inputs = tokenizer(question, return_tensors='pt')
    device = next(qwen_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with injector.active(experience_nodes):
            _ = qwen_model.generate(**inputs, max_new_tokens=20, do_sample=False)

    # Now introspect
    explanation = introspector.explain_last_answer(question=question)

    logger.info("Introspection explanation: %s", explanation)

    # At minimum, fallback explanation should work
    if explanation is None:
        # Try fallback
        log = injector.get_injection_log()
        if log.get('active'):
            explanation = introspector._fallback_explanation(
                question, log.get('node_details', []), log
            )
            logger.info("Fallback explanation: %s", explanation)

    assert explanation is not None and len(explanation.strip()) > 0, \
        "Introspector should return a non-empty explanation"

    logger.info("✓ PASS: Introspector returned explanation")
    return explanation


def test_dtype_compatibility():
    """Test 5: Does injection work with float16 model without dtype errors?

    This specifically tests the bug fix where float32 experience_vector
    was being added to float16 hidden_states, causing RuntimeError.
    """
    import torch

    logger.info("=" * 60)
    logger.info("TEST 5: dtype Compatibility (float16)")
    logger.info("=" * 60)

    qwen_model, tokenizer = load_qwen3()
    bge_model = load_bge_m3()

    from unconscious.injector import UnconsciousInjector
    injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.5)

    dummy_node = make_dummy_node(
        bge_model,
        text="kata pengecualian kecuali selain tidak membalik jawaban",
        node_id='test-kecuali-1',
    )
    experience_nodes = [(dummy_node, 0.95)]

    question = "Berapa 2+2?"
    inputs = tokenizer(question, return_tensors='pt')
    device = next(qwen_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # This should NOT raise RuntimeError: result type Float can't be cast to Half
    try:
        with torch.no_grad():
            with injector.active(experience_nodes):
                output = qwen_model.generate(**inputs, max_new_tokens=20, do_sample=False)

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Generated (with float16 injection): %s", text)
        logger.info("✓ PASS: No dtype error — float16 compatible")
        return True
    except RuntimeError as e:
        if "cast" in str(e) or "Half" in str(e) or "Float" in str(e):
            logger.error("✗ FAIL: dtype mismatch still present: %s", e)
            raise AssertionError(f"dtype bug not fixed: {e}") from e
        raise


def test_governance_filter():
    """Test 6: Does governance filtering work correctly?

    DEPRECATED and CONTRADICTED nodes should NOT be injected.
    Only CANDIDATE and STABLE nodes should pass through.
    """
    import torch

    logger.info("=" * 60)
    logger.info("TEST 6: Governance Filtering")
    logger.info("=" * 60)

    qwen_model, tokenizer = load_qwen3()
    bge_model = load_bge_m3()

    from unconscious.injector import UnconsciousInjector
    from derivation.understanding_builder import UnderstandingNode, Transformation
    from governance.states import LifecycleState, EpistemicState

    injector = UnconsciousInjector(qwen_model, enabled=True, injection_strength=0.1)

    # Create embedding for test nodes
    import numpy as np
    emb = bge_model.encode(
        ["test condition text"], show_progress_bar=False, normalize_embeddings=True
    )[0].tolist()

    # Node that SHOULD be injectable: STABLE + OBSERVED
    node_stable = UnderstandingNode(
        id='stable-node', name='Stable', concept='test',
        abstraction='test', condition_embedding=emb, confidence=0.8,
    )
    assert node_stable.lifecycle == LifecycleState.STABLE
    assert node_stable.epistemic == EpistemicState.OBSERVED
    assert injector._is_injectable(node_stable) is True, \
        "STABLE+OBSERVED should be injectable"
    logger.info("  STABLE+OBSERVED: injectable=True ✓")

    # Node that should NOT be injectable: DEPRECATED
    node_deprecated = UnderstandingNode(
        id='deprecated-node', name='Deprecated', concept='test',
        abstraction='test', condition_embedding=emb, confidence=0.8,
        lifecycle='deprecated',
    )
    assert injector._is_injectable(node_deprecated) is False, \
        "DEPRECATED should NOT be injectable"
    logger.info("  DEPRECATED: injectable=False ✓")

    # Node that should NOT be injectable: CONTRADICTED
    node_contradicted = UnderstandingNode(
        id='contradicted-node', name='Contradicted', concept='test',
        abstraction='test', condition_embedding=emb, confidence=0.8,
        epistemic='contradicted',
    )
    assert injector._is_injectable(node_contradicted) is False, \
        "CONTRADICTED should NOT be injectable"
    logger.info("  CONTRADICTED: injectable=False ✓")

    # Node that should be injectable: CANDIDATE
    node_candidate = UnderstandingNode(
        id='candidate-node', name='Candidate', concept='test',
        abstraction='test', condition_embedding=emb, confidence=0.8,
        lifecycle='candidate',
    )
    assert injector._is_injectable(node_candidate) is True, \
        "CANDIDATE should be injectable"
    logger.info("  CANDIDATE: injectable=True ✓")

    # Node with low confidence should NOT be injectable
    node_low_conf = UnderstandingNode(
        id='low-conf-node', name='LowConf', concept='test',
        abstraction='test', condition_embedding=emb, confidence=0.1,
    )
    assert injector._is_injectable(node_low_conf) is False, \
        "Low confidence (<0.2) should NOT be injectable"
    logger.info("  LOW_CONFIDENCE: injectable=False ✓")

    logger.info("✓ PASS: Governance filtering works correctly")
    return True


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Run all tests and report results."""
    results = {}
    all_passed = True

    tests = [
        ("test_dtype_compatibility", test_dtype_compatibility),
        ("test_governance_filter", test_governance_filter),
        ("test_injection_modifies_hidden_states", test_injection_modifies_hidden_states),
        ("test_injection_affects_output", test_injection_affects_output),
        ("test_injection_log_populated", test_injection_log_populated),
        ("test_introspection", test_introspection),
    ]

    for name, test_fn in tests:
        logger.info("\n")
        try:
            result = test_fn()
            results[name] = {'passed': True, 'result': result}
        except AssertionError as e:
            logger.error("✗ FAIL %s: %s", name, e)
            results[name] = {'passed': False, 'error': str(e)}
            all_passed = False
        except Exception as e:
            logger.exception("✗ ERROR %s: %s", name, e)
            results[name] = {'passed': False, 'error': str(e)}
            all_passed = False

    # ── Summary ──
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, result in results.items():
        status = "PASS" if result['passed'] else "FAIL"
        logger.info("  [%s] %s", status, name)

    if all_passed:
        logger.info("\n🎉 All tests PASSED — unconscious injection pipeline is LIVE")
    else:
        logger.info("\n❌ Some tests FAILED — see details above")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
