#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_adversarial.py
# @WHAT:  Adversarial test suite — questions designed to TRICK the system
# @PART:  self-ai/tests
# @ENTRY: python tests/test_adversarial.py

"""SELF-AI v17 Adversarial Test Suite

These tests are deliberately designed to TRICK the system into giving
wrong answers. Each test targets a specific cognitive blind spot:

1. Double Negative  — Stacked negation that flips meaning
2. Misleading Context — Contrast markers that flip meaning
3. Distractor Numbers — Extra numbers that shouldn't be used
4. Trap Questions — Popular opinion contradicted by text
5. Word-Sense Disambiguation Trap — Same word, different meaning
6. Quantitative Trick — Multi-step arithmetic with a common pitfall
7. Irrelevant Context — Long text with answer hidden in one sentence
8. Emotional vs Factual — Text has strong emotions but question asks for fact

For each test we define:
  - text:        The input passage
  - question:    The question to answer
  - wrong_answer: What the system should NOT say (the trap)
  - expected_keywords: What the correct answer should contain
  - explanation: Why this is a trick

SUCCESS = wrong answer NOT in output AND expected keyword IS in output
FAILURE = system fell for the trap (output contains wrong_answer)

Some tests will fail — that's the POINT. We identify weaknesses to fix.
"""
import sys
import os
import time
import json

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from derivation.text_comprehension import TextComprehension
from derivation.engine import DerivationEngine
from core.self import SelfCore


# ═══════════════════════════════════════════════════════════════
# ADVERSARIAL TEST CASES
# ═══════════════════════════════════════════════════════════════

ADVERSARIAL_TESTS = []

# ───────────────────────────────────────────────────────────────
# 1. DOUBLE NEGATIVE — Stacked negation
# "TIDAK BUKAN tidak benar" = triple negation = actually TRUE
# KECUALI means "except" — so the answer is the one that IS the exception
# i.e., the one that IS NOT true (the FALSE one)
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-01',
    'category': 'Double Negative',
    'text': (
        'Tiga pernyataan berikut tentang Burung Merak: '
        '(1) Burung Merak bisa terbang. '
        '(2) Burung Merak adalah hewan mamalia. '
        '(3) Burung Merak memiliki bulu yang indah. '
        'Semua pernyataan berikut TIDAK BUKAN tidak benar, KECUALI...'
    ),
    'question': 'Pernyataan mana yang tidak benar tentang Burung Merak?',
    'wrong_answer': 'bulu yang indah',    # Trap: pick something that IS true
    'expected_keywords': ['mamalia', '(2)', 'hewan mamalia'],  # Correct: (2) is false
    'explanation': (
        'Triple negation "TIDAK BUKAN tidak benar" cancels out to "benar" (true). '
        'So all statements are TRUE, EXCEPT one. The false one is (2): '
        'Merak is a bird, not a mammal. Trap: system reads "tidak benar" and '
        'picks the wrong statement.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 2. MISLEADING CONTEXT — Contrast markers flip meaning
# "TETAPI" (BUT) signals a contrast — the real situation is AFTER "tetapi"
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-02',
    'category': 'Misleading Context',
    'text': (
        'Raja Ramadhan sangat kaya dan berkuasa. '
        'Ia memiliki istana megah dan harta yang tak terhitung. '
        'TETAPI dia tidur di lantai dan makan seadanya setiap hari. '
        'Ia menyumbangkan seluruh kekayaannya untuk rakyat miskin.'
    ),
    'question': 'Bagaimana kehidupan Raja Ramadhan sehari-hari?',
    'wrong_answer': 'kaya',   # Trap: pick the first impression before "TETAPI"
    'expected_keywords': ['sederhana', 'seadanya', 'tidur di lantai', 'sederhana', 'menyumbang'],
    'explanation': (
        'The text first says "kaya dan berkuasa" but TETAPI flips it. '
        'The actual daily life is "tidur di lantai dan makan seadanya". '
        'Trap: system grabs the first descriptor "kaya" without processing '
        'the contrast marker.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 3. DISTRACTOR NUMBERS — Extra numbers that shouldn't be used
# Only Andi's marbles matter, not Budi's
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-03',
    'category': 'Distractor Numbers',
    'text': (
        'Andi punya 5 kelereng merah dan 3 kelereng biru. '
        'Budi punya 7 kelereng. '
        'Citra punya 2 kelereng emas.'
    ),
    'question': 'Berapa jumlah kelereng Andi?',
    'wrong_answer': '15',   # Trap: 5+3+7=15 (including Budi's)
    'expected_keywords': ['8'],
    'explanation': (
        'Only Andi\'s marbles should be counted: 5 + 3 = 8. '
        'Budi\'s 7 and Citra\'s 2 are distractors. '
        'Trap: system adds all numbers = 5+3+7+2 = 17 or 5+3+7 = 15.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 4. TRAP QUESTIONS — Popular opinion contradicted by text
# Everyone SAYS Lia is pemarah, but the text says she NEVER gets angry
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-04',
    'category': 'Trap Question',
    'text': (
        'Semua orang berkata Lia pemarah dan mudah tersinggung. '
        'Tapi Lia tidak pernah marah kepada siapapun. '
        'Ia selalu tersenyum dan sabar menghadapi semua cemoohan.'
    ),
    'question': 'Sifat Lia yang sebenarnya berdasarkan cerita?',
    'wrong_answer': 'pemarah',   # Trap: everyone says she's pemarah
    'expected_keywords': ['sabar', 'tidak pemarah', 'tidak pernah marah'],
    'explanation': (
        'Popular opinion says "pemarah" but the FACT is "tidak pernah marah" '
        'and "selalu tersenyum dan sabar". Trap: system picks the popular '
        'opinion instead of the factual evidence in the text.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 5. WORD-SENSE DISAMBIGUATION TRAP — Same word, different meaning
# "Kepala" in "kepala desa" = leader, "kepala ikan" = body part
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-05',
    'category': 'Word-Sense Disambiguation',
    'text': (
        'Kepala desa itu bijaksana dan adil. '
        'Ia memimpin dengan penuh tanggung jawab. '
        'Kepala ikan itu dibuang ke tempat sampah.'
    ),
    'question': 'Kata "kepala" pada kalimat pertama bermakna?',
    'wrong_answer': 'bagian tubuh',   # Trap: literal meaning of "kepala"
    'expected_keywords': ['pemimpin', 'ketua', 'panglima', 'penguasa'],
    'explanation': (
        'In "kepala desa", "kepala" means "pemimpin" (leader), not "bagian tubuh" '
        '(body part). In "kepala ikan", it IS the body part. '
        'Trap: system picks the more common/literal meaning.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 6. QUANTITATIVE TRICK — Multi-step arithmetic with common pitfall
# Rina buys 3 books at Rp5,000 each, pays Rp20,000
# Change = 20000 - (3 * 5000) = 5000, NOT 15000 (forgetting to multiply)
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-06',
    'category': 'Quantitative Trick',
    'text': (
        'Rina membeli 3 buku seharga Rp5.000 setiap buku. '
        'Ia membayar dengan uang Rp20.000.'
    ),
    'question': 'Berapa kembalian Rina?',
    'wrong_answer': '15000',   # Trap: 20000 - 5000 = 15000 (forgot to multiply)
    'expected_keywords': ['5000', '5.000'],
    'explanation': (
        '3 books x Rp5.000 = Rp15.000 total cost. '
        'Rp20.000 - Rp15.000 = Rp5.000 change. '
        'Trap: system subtracts 5000 from 20000 directly = 15000, '
        'forgetting to multiply first.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 7. IRRELEVANT CONTEXT — Long text with answer hidden in one sentence
# Lots of details, but the answer is only in one specific sentence
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-07',
    'category': 'Irrelevant Context',
    'text': (
        'Jakarta adalah ibu kota Indonesia dengan penduduk lebih dari 10 juta jiwa. '
        'Kota ini memiliki monumen nasional yang terkenal. '
        'Lalu lintas Jakarta sangat padat setiap hari. '
        'Banyak gedung pencakar langit di pusat kota. '
        'Musim hujan di Jakarta sering menyebabkan banjir. '
        'Presiden Indonesia saat ini berkantor di Istana Merdeka yang terletak di Jakarta. '
        'Makanan khas Jakarta adalah kerak telor. '
        'Jakarta memiliki tim sepak bola bernama Persija.'
    ),
    'question': 'Makanan khas Jakarta menurut teks tersebut?',
    'wrong_answer': 'nasi goreng',   # Trap: guess a common Jakarta food not in text
    'expected_keywords': ['kerak telor'],
    'explanation': (
        'The text mentions "kerak telor" once among 8 sentences of irrelevant detail. '
        'Trap: system guesses a common Jakarta food or picks up wrong details '
        'from the surrounding context instead of the exact text.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 8. EMOTIONAL VS FACTUAL — Text has strong emotions but question asks for fact
# The text is emotionally charged but the question asks for WHAT was cooked
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-08',
    'category': 'Emotional vs Factual',
    'text': (
        'Dengan hati yang berduka dan air mata mengalir di pipi, '
        'ibu memasak nasi putih dan lauk sayur untuk makan malam keluarga. '
        'Meskipun sedih, ibu tetap melayani anak-anaknya dengan penuh kasih sayang.'
    ),
    'question': 'Apa yang ibu masak untuk makan malam?',
    'wrong_answer': 'sedih',   # Trap: answer with the emotion instead of the fact
    'expected_keywords': ['nasi putih', 'lauk sayur', 'sayur'],
    'explanation': (
        'The text is emotionally charged ("berduka", "air mata", "sedih") '
        'but the question asks for a FACT ("apa yang ibu masak"). '
        'Correct answer: "nasi putih dan lauk sayur". '
        'Trap: system picks up the dominant emotion "sedih" as the answer.'
    ),
})

# ───────────────────────────────────────────────────────────────
# 9. ADDITIONAL: NEGATION SCOPE — "tidak" applies to the second clause only
# "X bukan Y, tapi Z" means X IS Z, NOT Y
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-09',
    'category': 'Negation Scope',
    'text': (
        'Siti bukan anak yang malas, tapi anak yang sangat rajin. '
        'Setiap pagi ia bangun jam 5 untuk membantu ibunya.'
    ),
    'question': 'Sifat Siti menurut cerita?',
    'wrong_answer': 'malas',   # Trap: "malas" appears before "bukan" negates it
    'expected_keywords': ['rajin', 'sangat rajin'],
    'explanation': (
        '"bukan anak yang malas" means she is NOT lazy. '
        '"tapi anak yang sangat rajin" means she IS very diligent. '
        'Trap: system sees "malas" and answers "malas" without '
        'processing the negation "bukan".'
    ),
})

# ───────────────────────────────────────────────────────────────
# 10. ADDITIONAL: COMPARATIVE TRAP — "lebih" with wrong reference
# ───────────────────────────────────────────────────────────────

ADVERSARIAL_TESTS.append({
    'id': 'ADV-10',
    'category': 'Comparative Trap',
    'text': (
        'Tinggi Andi 150 cm. Tinggi Budi 160 cm. '
        'Andi lebih pendek dari Budi.'
    ),
    'question': 'Siapa yang lebih tinggi?',
    'wrong_answer': 'andi',   # Trap: answer the one mentioned first
    'expected_keywords': ['budi'],
    'explanation': (
        'Budi is 160 cm, Andi is 150 cm. Budi is taller. '
        'Trap: system picks "Andi" because he\'s mentioned first '
        'or confuses "lebih pendek" direction.'
    ),
})


# ═══════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════

def run_adversarial_tests():
    """Run all adversarial tests and report results."""
    self_core = SelfCore()
    tc = TextComprehension(self_core, use_embeddings=False, use_llm=False)
    engine = DerivationEngine(self_core)

    print("=" * 70)
    print("SELF-AI v17 — ADVERSARIAL TEST SUITE")
    print("Questions designed to TRICK the system — identify weaknesses")
    print("=" * 70)

    results = []
    passed_trap = 0   # System did NOT output the wrong answer
    failed_trap = 0   # System FELL for the trap
    passed_expected = 0  # System output contains expected keyword
    total = len(ADVERSARIAL_TESTS)
    start_time = time.time()

    for test in ADVERSARIAL_TESTS:
        test_id = test['id']
        category = test['category']
        text = test['text']
        question = test['question']
        wrong_answer = test['wrong_answer']
        expected_keywords = test['expected_keywords']
        explanation = test['explanation']

        print(f"\n{'─' * 60}")
        print(f"  [{test_id}] {category}")
        print(f"  Q: {question}")
        print(f"  Trap answer (WRONG): {wrong_answer}")
        print(f"  Expected keywords: {expected_keywords}")

        # ── Route question through appropriate engine ──
        # Use DerivationEngine for quantitative ("berapa") questions
        # Use TextComprehension directly for qualitative questions
        if 'berapa' in question.lower() and any(c.isdigit() for c in text):
            result = engine.derive_from_text(text, question)
        else:
            result = tc.comprehend(text, question)

        answer = result.get('answer')
        confidence = result.get('confidence', 0.0)
        method = result.get('method', 'unknown')
        q_type = result.get('question_type', 'N/A')

        answer_str = str(answer).lower() if answer is not None else ''

        # ── Check 1: Did system fall for the trap? ──
        wrong_lower = wrong_answer.lower()
        trap_triggered = wrong_lower in answer_str

        # ── Check 2: Does answer contain expected keywords? ──
        keyword_found = False
        matched_keyword = None
        if answer is not None:
            for kw in expected_keywords:
                if kw.lower() in answer_str:
                    keyword_found = True
                    matched_keyword = kw
                    break

        # ── Determine result ──
        if trap_triggered:
            # FELL FOR THE TRAP — worst outcome
            status = "TRAPPED"
            status_icon = "❌"
            failed_trap += 1
        elif keyword_found:
            # Got the right answer
            status = "PASS"
            status_icon = "✅"
            passed_trap += 1
            passed_expected += 1
        elif answer is None:
            # No answer — not trapped, but not right either
            status = "NO ANSWER"
            status_icon = "⚠️"
            passed_trap += 1  # Didn't fall for trap at least
        else:
            # Some other answer — not trapped, but not matching expected
            status = "UNEXPECTED"
            status_icon = "🔶"
            passed_trap += 1  # Didn't fall for trap

        result_entry = {
            'id': test_id,
            'category': category,
            'status': status,
            'trap_triggered': trap_triggered,
            'keyword_found': keyword_found,
            'matched_keyword': matched_keyword,
            'answer': str(answer) if answer is not None else None,
            'confidence': confidence,
            'method': method,
            'question_type': q_type,
        }
        results.append(result_entry)

        # ── Print detailed result ──
        print(f"\n  {status_icon} STATUS: {status}")
        print(f"  Answer: {answer} (confidence={confidence:.2f}, method={method}, q_type={q_type})")
        if trap_triggered:
            print(f"  🚨 TRAP TRIGGERED: '{wrong_answer}' found in output!")
        if keyword_found:
            print(f"  ✓ Expected keyword '{matched_keyword}' found in answer")
        if not keyword_found and not trap_triggered:
            print(f"  ⚠ No expected keyword found in answer")
            print(f"     Expected any of: {expected_keywords}")
        print(f"  💡 {explanation[:100]}...")

    elapsed = time.time() - start_time

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ADVERSARIAL TEST SUMMARY")
    print("=" * 70)
    print(f"  Total tests:        {total}")
    print(f"  Trap NOT triggered: {passed_trap}/{total} ({passed_trap/total*100:.0f}%)")
    print(f"  Trap TRIGGERED:     {failed_trap}/{total} ({failed_trap/total*100:.0f}%)")
    print(f"  Expected keyword:   {passed_expected}/{total} ({passed_expected/total*100:.0f}%)")
    print(f"  Elapsed:            {elapsed:.2f}s")

    # Category breakdown
    print("\nBREAKDOWN PER CATEGORY:")
    categories = {}
    for test in ADVERSARIAL_TESTS:
        cat = test['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'trapped': 0, 'pass': 0, 'unexpected': 0, 'no_answer': 0}
        categories[cat]['total'] += 1

    for r in results:
        cat = next(t['category'] for t in ADVERSARIAL_TESTS if t['id'] == r['id'])
        if r['trap_triggered']:
            categories[cat]['trapped'] += 1
        elif r['status'] == 'PASS':
            categories[cat]['pass'] += 1
        elif r['status'] == 'NO ANSWER':
            categories[cat]['no_answer'] += 1
        else:
            categories[cat]['unexpected'] += 1

    for cat, counts in categories.items():
        if counts['trapped'] > 0:
            status_icon = "❌"
        elif counts['pass'] == counts['total']:
            status_icon = "✅"
        else:
            status_icon = "🔶"
        trapped_str = f" 🚨{counts['trapped']} trapped" if counts['trapped'] > 0 else ""
        print(f"  {status_icon} {cat}: {counts['pass']}/{counts['total']} pass{trapped_str}"
              f" ({counts['unexpected']} unexpected, {counts['no_answer']} no answer)")

    # ── Detailed failure report ──
    trapped_tests = [r for r in results if r['trap_triggered']]
    if trapped_tests:
        print("\n" + "=" * 70)
        print("🚨 TRAPS THAT WERE TRIGGERED — NEEDS FIXING:")
        print("=" * 70)
        for r in trapped_tests:
            test = next(t for t in ADVERSARIAL_TESTS if t['id'] == r['id'])
            print(f"\n  {r['id']} [{r['category']}]: {test['question']}")
            print(f"    Trap: system said '{test['wrong_answer']}' (WRONG!)")
            print(f"    Got:  {r['answer']}")
            print(f"    Expected: {test['expected_keywords']}")
            print(f"    Fix: {test['explanation'][:150]}...")

    # ── Unexpected answers (not trapped, but not matching expected either) ──
    unexpected_tests = [r for r in results if r['status'] == 'UNEXPECTED']
    if unexpected_tests:
        print("\n" + "=" * 70)
        print("🔶 UNEXPECTED ANSWERS — Not trapped, but not matching expected keywords:")
        print("=" * 70)
        for r in unexpected_tests:
            test = next(t for t in ADVERSARIAL_TESTS if t['id'] == r['id'])
            print(f"\n  {r['id']} [{r['category']}]: {test['question']}")
            print(f"    Got:      {r['answer']}")
            print(f"    Expected: {test['expected_keywords']}")
            print(f"    Trap not triggered (good!), but answer doesn't match expected keywords")

    # ── Save results ──
    benchmark_data = {
        'version': 'v17',
        'test_type': 'adversarial',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total': total,
        'trap_not_triggered': passed_trap,
        'trap_triggered': failed_trap,
        'expected_keyword_found': passed_expected,
        'elapsed': elapsed,
        'results': results,
        'categories': {cat: f"{counts['pass']}/{counts['total']} (trapped: {counts['trapped']})" 
                       for cat, counts in categories.items()},
    }

    benchmark_path = os.path.join(PROJECT_ROOT, 'benchmark', 'adversarial_results.json')
    os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2, default=str)
    print(f"\nBenchmark saved to: {benchmark_path}")

    return passed_trap, passed_expected, total, results


if __name__ == '__main__':
    passed_trap, passed_expected, total, results = run_adversarial_tests()
    # Exit code: 0 if no traps triggered, 1 if any trap was triggered
    sys.exit(1 if (total - passed_trap) > 0 else 0)
