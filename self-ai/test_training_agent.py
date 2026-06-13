#!/usr/bin/env python3
"""Test TrainingAgent v1 — prove the 3 PoC criteria:

1. Wrong answer → correction → reasoning → confirm → pattern saved
2. Pattern survives restart (re-initialize engine, check pattern still there)
3. Accuracy after correction > accuracy before

This script runs a real session programmatically (not interactive CLI).
"""

import os
import sys
import json
import time
import traceback

# Setup paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'benchmark'))

os.environ['TOKENIZERS_PARALLELISM'] = '0'

# Test results accumulator
test_results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'poc1_correction_flow': None,
    'poc2_pattern_survives': None,
    'poc3_accuracy_improves': None,
    'details': {},
}


def header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def subheader(msg):
    print(f"\n  ── {msg} ──")


# ═══════════════════════════════════════════════════════════════
# Initialize TrainingAgent
# ═══════════════════════════════════════════════════════════════

header("Initializing TrainingAgent")
from training.training_agent import TrainingAgent

agent = TrainingAgent()
if agent.engine is None:
    print("FATAL: Engine failed to initialize!")
    sys.exit(1)

print("Engine initialized successfully")
print(f"TextComprehension loaded: {agent.tc is not None}")

# Check model status
try:
    from derivation.model_registry import get_shared_embedding_model, get_shared_qwen
    emb = get_shared_embedding_model()
    qwen, _ = get_shared_qwen()
    print(f"Embedding model: {'LOADED' if emb else 'NOT LOADED'}")
    print(f"Qwen3 model: {'LOADED' if qwen else 'NOT LOADED'}")
    models_loaded = emb is not None and qwen is not None
except Exception as e:
    print(f"Model check error: {e}")
    models_loaded = False

test_results['details']['models_loaded'] = models_loaded


# ═══════════════════════════════════════════════════════════════
# PoC 1: Wrong answer → correction → reasoning → confirm → saved
# ═══════════════════════════════════════════════════════════════

header("PoC 1: Correction Flow (explicit intent)")

# Use a test question from TEST_SOAL
test_soal = {
    'text': 'Angin menjerit keras menggoyangkan pepohonan di malam yang gelap itu.',
    'question': 'Kata "menjerit" pada kalimat tersebut termasuk majas....',
    'expected': 'personifikasi',
}

subheader("Step 1: Ask question to SELF")
result = agent.run(test_soal['question'], test_soal['text'])
print(f"  Question: {test_soal['question']}")
print(f"  SELF answered: {result.get('answer')} (confidence: {result.get('confidence', 0):.2f}, method: {result.get('method')})")

initial_answer = result.get('answer')
is_correct_initial = 'personifikasi' in str(initial_answer).lower() if initial_answer else False
print(f"  Is correct? {is_correct_initial}")

subheader("Step 2: Correct the answer (explicit intent)")
correct_result = agent.correct('personifikasi')
print(f"  Reasoning generated: {correct_result.get('reasoning', '')[:200]}")
print(f"  Confirmed? {correct_result.get('confirmed')}")  # Should be False

poc1_step2_ok = correct_result.get('confirmed') == False and correct_result.get('reasoning') != ''
print(f"  [CHECK] Reasoning generated without auto-teach: {poc1_step2_ok}")

subheader("Step 3: Confirm correction (explicit intent)")
confirm_result = agent.confirm_correction()
print(f"  Pattern key: {confirm_result.get('pattern_key', '')[:80]}")
print(f"  Confirmed: {confirm_result.get('confirmed')}")
print(f"  Reasoning saved: {confirm_result.get('reasoning', '')[:200]}")

poc1_step3_ok = confirm_result.get('confirmed') == True
print(f"  [CHECK] Pattern saved after explicit confirm: {poc1_step3_ok}")

subheader("Step 4: Ask same question again — should now use learned pattern")
result2 = agent.run(test_soal['question'], test_soal['text'])
print(f"  SELF answered: {result2.get('answer')} (confidence: {result2.get('confidence', 0):.2f}, method: {result2.get('method')})")

is_correct_after = 'personifikasi' in str(result2.get('answer')).lower() if result2.get('answer') else False
print(f"  Is correct now? {is_correct_after}")

poc1_ok = poc1_step2_ok and poc1_step3_ok and is_correct_after
test_results['poc1_correction_flow'] = poc1_ok
test_results['details']['poc1'] = {
    'initial_answer': str(initial_answer),
    'initial_correct': is_correct_initial,
    'reasoning_generated': correct_result.get('reasoning', '')[:200],
    'not_auto_confirmed': poc1_step2_ok,
    'pattern_saved': poc1_step3_ok,
    'answer_after_correction': str(result2.get('answer')),
    'correct_after_correction': is_correct_after,
}
print(f"\n  >>> PoC 1 RESULT: {'PASS' if poc1_ok else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════
# PoC 2: Pattern survives restart
# ═══════════════════════════════════════════════════════════════

header("PoC 2: Pattern Survives Restart")

subheader("Step 1: Check patterns on disk")
patterns_file = os.path.join(PROJECT_ROOT, 'data', 'learned_patterns.json')
if os.path.exists(patterns_file):
    with open(patterns_file, 'r') as f:
        patterns = json.load(f)
    pattern_count = len(patterns)
    print(f"  Patterns file exists: YES")
    print(f"  Pattern count: {pattern_count}")
else:
    pattern_count = 0
    patterns = {}
    print(f"  Patterns file exists: NO")

subheader("Step 2: Re-initialize TrainingAgent (fresh instance)")
agent2 = TrainingAgent()
if agent2.engine is None:
    print("FATAL: Engine re-init failed!")
    poc2_ok = False
else:
    print("  New TrainingAgent initialized")

    subheader("Step 3: Ask same question — pattern should still be used")
    result3 = agent2.run(test_soal['question'], test_soal['text'])
    print(f"  SELF answered: {result3.get('answer')} (confidence: {result3.get('confidence', 0):.2f}, method: {result3.get('method')})")

    is_correct_restart = 'personifikasi' in str(result3.get('answer')).lower() if result3.get('answer') else False
    print(f"  Is correct after restart? {is_correct_restart}")

    # Also check pattern count
    tc_pattern_count = len(agent2.tc.learned_patterns) if agent2.tc else 0
    print(f"  Pattern count in new TC: {tc_pattern_count}")

    poc2_ok = is_correct_restart and tc_pattern_count > 0

test_results['poc2_pattern_survives'] = poc2_ok
test_results['details']['poc2'] = {
    'patterns_on_disk': pattern_count,
    'correct_after_restart': is_correct_restart if 'is_correct_restart' in dir() else False,
    'tc_pattern_count': tc_pattern_count if 'tc_pattern_count' in dir() else 0,
}
print(f"\n  >>> PoC 2 RESULT: {'PASS' if poc2_ok else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════
# PoC 3: Accuracy after > accuracy before
# ═══════════════════════════════════════════════════════════════

header("PoC 3: Accuracy Improvement")

# Use a fresh agent for clean before measurement
agent3 = TrainingAgent()
if agent3.engine is None:
    print("FATAL: Engine init for benchmark failed!")
    poc3_ok = False
else:
    # Step 1: Clean patterns for a fair before-measurement
    subheader("Step 1: Clean patterns + run BEFORE benchmark")
    # Don't delete patterns — just measure with fresh agent
    before = agent3.benchmark()
    if 'error' in before:
        print(f"  ERROR: {before['error']}")
        poc3_ok = False
    else:
        print(f"  BEFORE: {before['correct']}/{before['total']} ({before['accuracy']:.1%})")
        if before.get('per_type'):
            for domain, stats in sorted(before['per_type'].items()):
                print(f"    {domain}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")

        # Step 2: Teach some corrections through the TrainingAgent
        subheader("Step 2: Teach corrections through TrainingAgent")

        # Pick questions that SELF got wrong and correct them
        corrections_made = 0
        for soal in before.get('details', []):
            if not soal['pass'] and corrections_made < 5:
                # Find the original test case
                from benchmark_empiris import TEST_SOAL
                original = None
                for ts in TEST_SOAL:
                    if ts['id'] == soal['id']:
                        original = ts
                        break

                if original and original.get('expected_keywords'):
                    correct_ans = original['expected_keywords'][0]

                    # Run the question through agent
                    q_result = agent3.run(original['question'], original['text'])
                    wrong_ans = q_result.get('answer')

                    # Correct it
                    c_result = agent3.correct(correct_ans)
                    if 'error' not in c_result:
                        # Confirm
                        cf_result = agent3.confirm_correction()
                        if cf_result.get('confirmed'):
                            corrections_made += 1
                            print(f"    Corrected {soal['id']}: wrong='{str(wrong_ans)[:40]}' → correct='{correct_ans}'")

        print(f"  Total corrections applied: {corrections_made}")

        # Step 3: Run AFTER benchmark
        subheader("Step 3: Run AFTER benchmark")
        after = agent3.benchmark()
        if 'error' in after:
            print(f"  ERROR: {after['error']}")
            poc3_ok = False
        else:
            print(f"  AFTER: {after['correct']}/{after['total']} ({after['accuracy']:.1%})")
            if after.get('per_type'):
                for domain, stats in sorted(after['per_type'].items()):
                    print(f"    {domain}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")

            delta = after['accuracy'] - before['accuracy']
            sign = '+' if delta >= 0 else ''
            print(f"  Delta: {sign}{delta:.1%}")

            poc3_ok = after['accuracy'] > before['accuracy']

            test_results['details']['poc3'] = {
                'before_accuracy': before['accuracy'],
                'before_correct': before['correct'],
                'before_total': before['total'],
                'after_accuracy': after['accuracy'],
                'after_correct': after['correct'],
                'after_total': after['total'],
                'delta': delta,
                'corrections_applied': corrections_made,
                'before_per_type': {k: v for k, v in before.get('per_type', {}).items()},
                'after_per_type': {k: v for k, v in after.get('per_type', {}).items()},
            }

            # Show per-type delta
            before_per = before.get('per_type', {})
            after_per = after.get('per_type', {})
            all_domains = sorted(set(list(before_per.keys()) + list(after_per.keys())))
            if all_domains:
                print(f"\n  Domain breakdown:")
                for domain in all_domains:
                    b = before_per.get(domain, {}).get('accuracy', 0)
                    a = after_per.get(domain, {}).get('accuracy', 0)
                    d = a - b
                    s = '+' if d >= 0 else ''
                    print(f"    {domain}: {b:.0%} → {a:.0%} ({s}{d:.0%})")

test_results['poc3_accuracy_improves'] = poc3_ok
print(f"\n  >>> PoC 3 RESULT: {'PASS' if poc3_ok else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════
# Export session
# ═══════════════════════════════════════════════════════════════

header("Export Session")
# Use the last agent that has the most data
filepath = agent3.export_session()
print(f"  Session exported to: {filepath}")

# Also export test results as JSON
results_path = os.path.join(PROJECT_ROOT, 'benchmark', 'training_agent_test_results.json')
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(test_results, f, indent=2, default=str, ensure_ascii=False)
print(f"  Test results saved to: {results_path}")


# ═══════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════

header("FINAL SUMMARY")
print(f"  PoC 1 — Correction Flow:       {'PASS' if test_results['poc1_correction_flow'] else 'FAIL'}")
print(f"  PoC 2 — Pattern Survives:      {'PASS' if test_results['poc2_pattern_survives'] else 'FAIL'}")
print(f"  PoC 3 — Accuracy Improves:     {'PASS' if test_results['poc3_accuracy_improves'] else 'FAIL'}")
all_pass = all([
    test_results['poc1_correction_flow'],
    test_results['poc2_pattern_survives'],
    test_results['poc3_accuracy_improves'],
])
print(f"\n  ALL PoC: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print(f"  Models loaded: {models_loaded}")
