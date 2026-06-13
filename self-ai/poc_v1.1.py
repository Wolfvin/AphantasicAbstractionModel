#!/usr/bin/env python3
"""PoC Test — TrainingAgent v1.1 (dengan context bug fix)

Jalankan: cd self-ai && python3 poc_v1.1.py

Buktikan 3 PoC criteria:
  1. Wrong → correction → reasoning → confirm → pattern saved
  2. Pattern survives restart
  3. Accuracy after correction > accuracy before
"""

import os, sys, json, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'benchmark'))
os.environ['TOKENIZERS_PARALLELISM'] = '0'

SEP = '=' * 60

# ═══════════════════════════════════════════════════════
# 1. Init
# ═══════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TrainingAgent v1.1 — PoC Test")
print(SEP)

t0 = time.time()
print(f"\n[{time.strftime('%H:%M:%S')}] Initializing engine...", flush=True)
from training.training_agent import TrainingAgent
agent = TrainingAgent()

if agent.engine is None:
    print("FATAL: Engine failed to initialize!")
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] Engine OK. Patterns: {len(agent.tc.learned_patterns)}")

# Model check
try:
    from derivation.model_registry import get_shared_embedding_model, get_shared_qwen
    emb = get_shared_embedding_model()
    qwen, _ = get_shared_qwen()
    emb_ok = emb is not None
    qwen_ok = qwen is not None
except Exception as e:
    emb_ok = False
    qwen_ok = False
    print(f"Model check error: {e}")

print(f"[{time.strftime('%H:%M:%S')}] bge-m3: {'LOADED' if emb_ok else 'NOT LOADED'}")
print(f"[{time.strftime('%H:%M:%S')}] Qwen3:  {'LOADED' if qwen_ok else 'NOT LOADED (fallback reasoning)'}")

# ═══════════════════════════════════════════════════════
# PoC 1: Wrong → correct → reasoning → confirm → saved
# ═══════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  PoC 1: Correction Flow (explicit intent)")
print(SEP)

test_soal = {
    'text': 'Angin menjerit keras menggoyangkan pepohonan di malam yang gelap itu.',
    'question': 'Kata "menjerit" pada kalimat tersebut termasuk majas....',
    'expected': 'personifikasi',
}

print(f"\n  Step 1: Ask question to SELF")
result = agent.run(test_soal['question'], test_soal['text'])
initial_answer = result.get('answer')
is_correct_initial = 'personifikasi' in str(initial_answer).lower() if initial_answer else False
print(f"    Question: {test_soal['question']}")
print(f"    SELF answered: {initial_answer} (conf: {result.get('confidence',0):.2f}, method: {result.get('method','')})")
print(f"    Is correct? {is_correct_initial}")

# BUG FIX VERIFICATION
context_is_narrative = agent._last_context == test_soal['text']
context_is_not_question = agent._last_context != test_soal['question']
print(f"    [BUG FIX] _last_context == narrative text: {context_is_narrative}")
print(f"    [BUG FIX] _last_context != question:       {context_is_not_question}")

print(f"\n  Step 2: Correct the answer (explicit intent)")
correct_result = agent.correct('personifikasi')
poc1_step2_ok = correct_result.get('confirmed') == False and correct_result.get('reasoning') != ''
print(f"    Reasoning: {correct_result.get('reasoning', '')[:200]}")
print(f"    Confirmed? {correct_result.get('confirmed')} (should be False)")
print(f"    [CHECK] Reasoning generated without auto-teach: {poc1_step2_ok}")

print(f"\n  Step 3: Confirm correction (explicit intent)")
confirm_result = agent.confirm_correction()
poc1_step3_ok = confirm_result.get('confirmed') == True
print(f"    Confirmed: {confirm_result.get('confirmed')}")
print(f"    Pattern key: {confirm_result.get('pattern_key', '')[:80]}")

print(f"\n  Step 4: Ask same question again — should now use learned pattern")
result2 = agent.run(test_soal['question'], test_soal['text'])
is_correct_after = 'personifikasi' in str(result2.get('answer')).lower() if result2.get('answer') else False
print(f"    SELF answered: {result2.get('answer')} (conf: {result2.get('confidence',0):.2f}, method: {result2.get('method','')})")
print(f"    Is correct now? {is_correct_after}")

poc1_ok = poc1_step2_ok and poc1_step3_ok and is_correct_after
print(f"\n  >>> PoC 1: {'PASS' if poc1_ok else 'FAIL'}")
print(f"      Bug fix verified: {'YES' if (context_is_narrative and context_is_not_question) else 'NO'}")

# ═══════════════════════════════════════════════════════
# PoC 2: Pattern survives restart
# ═══════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  PoC 2: Pattern Survives Restart")
print(SEP)

print(f"\n  Step 1: Check patterns on disk")
patterns_file = os.path.join(PROJECT_ROOT, 'data', 'learned_patterns.json')
if os.path.exists(patterns_file):
    with open(patterns_file, 'r') as f:
        patterns = json.load(f)
    pattern_count = len(patterns)
    print(f"    Patterns on disk: {pattern_count}")
else:
    pattern_count = 0
    print(f"    Patterns file not found!")

print(f"\n  Step 2: Re-initialize TrainingAgent (fresh instance)")
agent2 = TrainingAgent()
if agent2.engine is None:
    print("    FATAL: Re-init failed!")
    poc2_ok = False
else:
    tc_pattern_count = len(agent2.tc.learned_patterns) if agent2.tc else 0
    print(f"    New TrainingAgent OK. Patterns in memory: {tc_pattern_count}")

    print(f"\n  Step 3: Ask same question — pattern should still be used")
    result3 = agent2.run(test_soal['question'], test_soal['text'])
    is_correct_restart = 'personifikasi' in str(result3.get('answer')).lower() if result3.get('answer') else False
    print(f"    SELF answered: {result3.get('answer')} (conf: {result3.get('confidence',0):.2f}, method: {result3.get('method','')})")
    print(f"    Is correct after restart? {is_correct_restart}")
    poc2_ok = is_correct_restart and tc_pattern_count > 0

print(f"\n  >>> PoC 2: {'PASS' if poc2_ok else 'FAIL'}")

# ═══════════════════════════════════════════════════════
# PoC 3: Accuracy after > accuracy before
# ═══════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  PoC 3: Accuracy Improvement")
print(SEP)

print(f"\n  Step 1: Run BEFORE benchmark")
agent3 = TrainingAgent()
before = agent3.benchmark()
if 'error' in before:
    print(f"    ERROR: {before['error']}")
    poc3_ok = False
else:
    print(f"    BEFORE: {before['correct']}/{before['total']} ({before['accuracy']:.1%})")
    for domain, stats in sorted(before.get('per_type', {}).items()):
        print(f"      {domain}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")

    print(f"\n  Step 2: Teach corrections for wrong answers (max 5)")
    from benchmark_empiris import TEST_SOAL
    corrections_made = 0
    for soal in before.get('details', []):
        if not soal['pass'] and corrections_made < 5:
            original = None
            for ts in TEST_SOAL:
                if ts['id'] == soal['id']:
                    original = ts
                    break
            if original and original.get('expected_keywords'):
                correct_ans = original['expected_keywords'][0]
                q_result = agent3.run(original['question'], original['text'])
                c_result = agent3.correct(correct_ans)
                if 'error' not in c_result:
                    cf_result = agent3.confirm_correction()
                    if cf_result.get('confirmed'):
                        corrections_made += 1
                        print(f"      Corrected {soal['id']}: '{str(q_result.get('answer',''))[:40]}' → '{correct_ans}'")
    print(f"    Total corrections applied: {corrections_made}")

    print(f"\n  Step 3: Run AFTER benchmark")
    after = agent3.benchmark()
    if 'error' in after:
        print(f"    ERROR: {after['error']}")
        poc3_ok = False
    else:
        print(f"    AFTER: {after['correct']}/{after['total']} ({after['accuracy']:.1%})")
        for domain, stats in sorted(after.get('per_type', {}).items()):
            print(f"      {domain}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")

        delta = after['accuracy'] - before['accuracy']
        sign = '+' if delta >= 0 else ''
        print(f"    Delta: {sign}{delta:.1%}")

        # Per-type delta
        before_per = before.get('per_type', {})
        after_per = after.get('per_type', {})
        all_domains = sorted(set(list(before_per.keys()) + list(after_per.keys())))
        if all_domains:
            print(f"\n    Domain breakdown:")
            for domain in all_domains:
                b = before_per.get(domain, {}).get('accuracy', 0)
                a = after_per.get(domain, {}).get('accuracy', 0)
                d = a - b
                s = '+' if d >= 0 else ''
                print(f"      {domain}: {b:.0%} → {a:.0%} ({s}{d:.0%})")

        poc3_ok = after['accuracy'] > before['accuracy']

print(f"\n  >>> PoC 3: {'PASS' if poc3_ok else 'FAIL'}")

# ═══════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  FINAL SUMMARY")
print(SEP)
print(f"  bge-m3:       {'LOADED' if emb_ok else 'NOT LOADED'}")
print(f"  Qwen3:        {'LOADED' if qwen_ok else 'NOT LOADED (fallback)'}")
print(f"  Bug fix:      {'VERIFIED' if (context_is_narrative and context_is_not_question) else 'NOT VERIFIED'}")
print(f"  PoC 1 Flow:   {'PASS' if poc1_ok else 'FAIL'}")
print(f"  PoC 2 Persist: {'PASS' if poc2_ok else 'FAIL'}")
print(f"  PoC 3 Accuracy: {'PASS' if poc3_ok else 'FAIL'}")
all_pass = all([poc1_ok, poc2_ok, poc3_ok])
print(f"\n  RESULT: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
print(f"  Time: {time.time()-t0:.1f}s")

# Save JSON results
results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'duration_s': round(time.time()-t0, 1),
    'models': {'bge_m3': emb_ok, 'qwen3': qwen_ok},
    'bug_fix_verified': context_is_narrative and context_is_not_question,
    'poc1_correction_flow': poc1_ok,
    'poc2_pattern_survives': poc2_ok,
    'poc3_accuracy_improves': poc3_ok if 'poc3_ok' in dir() else False,
}
if 'before' in dir() and 'after' in dir() and isinstance(before, dict) and isinstance(after, dict) and 'error' not in before and 'error' not in after:
    results['poc3_detail'] = {
        'before': f"{before['correct']}/{before['total']} ({before['accuracy']:.1%})",
        'after': f"{after['correct']}/{after['total']} ({after['accuracy']:.1%})",
        'delta': f"{after['accuracy'] - before['accuracy']:+.1%}",
        'corrections': corrections_made if 'corrections_made' in dir() else 0,
        'before_per_type': before.get('per_type', {}),
        'after_per_type': after.get('per_type', {}),
    }
out = os.path.join(PROJECT_ROOT, 'benchmark', 'poc_v1.1_results.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)
print(f"\n  Results saved: {out}")
