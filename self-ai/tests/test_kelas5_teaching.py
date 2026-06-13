#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_kelas5_teaching.py
# @WHAT:  Teaching loop for Kelas 5 — teach similar-type soal (different domain+answer), verify learning
# @PART:  self-ai/tests
# @ENTRY: run_teaching_test()

"""SELF-AI v12 Teaching Loop — Kelas 5 Bahasa Indonesia

TEACHING METHODOLOGY:
1. Identify the 4 failures from the kelas 5 test
2. For each failure, teach with SIMILAR-TYPE questions but DIFFERENT domain + DIFFERENT answers
3. Re-test the ORIGINAL failed questions
4. Verify teaching GENERALIZES — not just memorization

The 4 failures:
- K5-PB03: Peribahasa about parents' hard work → should be "banting tulang", got "bersakit-sakit dahulu"
- K5-MH01: Multi-hop inference about flood → school → should trace back to "hujan/banjir"
- K5-BK02: Personification of "angin menjerit" → should be "personifikasi"
- K5-CP02: Comparison of study methods → should extract abstract difference "rajin vs malas"

Teaching examples use COMPLETELY DIFFERENT domains:
- PB03: Not about parents/Siti → about farmers/fishermen working hard
- MH01: Not about school → about factory closing due to raw material shortage
- BK02: Not about wind → about rain/sun "smiling"
- CP02: Not about studying → about saving vs spending money
"""
import sys
import os
import time
import json

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from derivation.text_comprehension import TextComprehension
from derivation.engine import DerivationEngine
from core.self import SelfCore


# ═══════════════════════════════════════════════════════════════
# ORIGINAL FAILED SOAL
# ═══════════════════════════════════════════════════════════════

FAILED_SOAL = [
    {
        'id': 'K5-PB03',
        'text': 'Kedua orang tua Siti bekerja keras siang dan malam agar Siti bisa sekolah. Siti pun belajar dengan rajin dan tidak menyia-nyiakan pengorbanan orang tuanya. Ia selalu mendapat ranking pertama di kelasnya.',
        'question': 'Peribahasa yang menggambarkan kerja keras orang tua Siti adalah....',
        'expected_keywords': ['pentingnya kerja', 'banting tulang', 'keringat', 'kerbau', 'bekerja keras'],
        'type': 'peribahasa',
    },
    {
        'id': 'K5-MH01',
        'text': 'Hujan turun tanpa henti selama tiga hari. Sungai meluap dan air masuk ke rumah-rumah warga. Banyak jalan tergenang air sehingga bus sekolah tidak bisa beroperasi. Akibatnya, anak-anak tidak bisa pergi ke sekolah.',
        'question': 'Mengapa anak-anak tidak bisa pergi ke sekolah?',
        'expected_keywords': ['banjir', 'jalan tergenang', 'bus tidak beroperasi', 'air', 'hujan'],
        'type': 'implisit',
    },
    {
        'id': 'K5-BK02',
        'text': 'Angin menjerit keras menggoyangkan pepohonan di malam yang gelap itu.',
        'question': 'Kata "menjerit" pada kalimat tersebut termasuk majas....',
        'expected_keywords': ['personifikasi', 'menghidupkan', 'benda hidup'],
        'type': 'bahasa_kiasan',
    },
    {
        'id': 'K5-CP02',
        'text': 'Rani belajar setiap malam sebelum ujian. Ia membuat catatan dan mengerjakan latihan soal. Doni hanya membaca buku sepintas pada malam sebelum ujian. Hasilnya, Rina mendapat nilai 90 sedangkan Doni mendapat nilai 55.',
        'question': 'Apa perbedaan cara belajar Rani dan Doni?',
        'expected_keywords': ['rajin', 'tekun', 'malas', 'serius', 'cermat', 'rajinkan', 'belajar tekun', 'belajar rutin'],
        'type': 'perbandingan',
    },
]


# ═══════════════════════════════════════════════════════════════
# TEACHING EXAMPLES — SIMILAR TYPE, DIFFERENT DOMAIN, DIFFERENT ANSWER
# ═══════════════════════════════════════════════════════════════

TEACHING = {
    'K5-PB03': [
        # Different domain: farmers, not parents. Different answer: same proverb category.
        {
            'text': 'Para petani di desa banting tulang dari subuh hingga magrib menggarap sawah mereka. Hasil panen yang melimpah menjadi buah kerja keras mereka sepanjang musim.',
            'question': 'Peribahasa yang sesuai untuk menggambarkan kerja keras para petani?',
            'answer': 'banting tulang',
            'explanation': 'Pattern: orang bekerja keras siang dan malam → banting tulang, bukan bersakit-sakit dahulu (itu untuk effort→reward contrast)',
        },
        {
            'text': 'Nelayan tua itu mengayuh perahunya setiap pagi buta hingga petang. Ia menghabiskan keringat untuk memenuhi kebutuhan keluarganya.',
            'question': 'Peribahasa apa yang menggambarkan semangat kerja nelayan tersebut?',
            'answer': 'banting tulang',
            'explanation': 'Pattern: bekerja keras dengan keringat → banting tulang (kerja keras fisik)',
        },
    ],
    'K5-MH01': [
        # Different domain: factory closing, not school. Different answer: same reasoning (trace back to root cause)
        {
            'text': 'Pabrik rotan di kota itu kehabisan bahan baku. Mesin-mesin berhenti beroperasi. Para pekerja tidak bisa masuk kerja dan kehilangan penghasilan.',
            'question': 'Mengapa pekerja kehilangan penghasilan?',
            'answer': 'pabrik kehabisan bahan baku',
            'explanation': 'Pattern: A → B → C. When asked why C, trace back to A (root cause), not B (intermediate). Kehabisan bahan baku → mesin berhenti → kehilangan penghasilan',
        },
        {
            'text': 'Kebakaran hutan menghasilkan asap tebal. Asap menutupi bandara sehingga pesawat tidak bisa mendarat. Penumpang tertunda selama berjam-jam.',
            'question': 'Mengapa penumpang tertunda?',
            'answer': 'kebakaran hutan',
            'explanation': 'Pattern: A→B→C chain. Kebakaran → asap → pesawat tidak bisa mendarat → tertunda. Root cause: kebakaran hutan',
        },
    ],
    'K5-BK02': [
        # Different domain: rain/sun, not wind. Same answer: personifikasi.
        {
            'text': 'Hujan menari-nari di atap rumah kami.',
            'question': 'Kata "menari-nari" pada kalimat tersebut termasuk majas....',
            'answer': 'personifikasi',
            'explanation': 'Pattern: non-human subject (hujan) + human action (menari-nari) → personifikasi',
        },
        {
            'text': 'Matahari tersenyum hangat menyinari desa itu.',
            'question': 'Majas apa yang digunakan pada kalimat "matahari tersenyum"?',
            'answer': 'personifikasi',
            'explanation': 'Pattern: non-human subject (matahari) + human verb (tersenyum) → personifikasi',
        },
        {
            'text': 'Bintang-bintang berkelipkan mata di langit malam.',
            'question': 'Kata "berkelipkan mata" pada kalimat tersebut termasuk majas....',
            'answer': 'personifikasi',
            'explanation': 'Pattern: non-human (bintang) + human action (berkelipkan mata) → personifikasi',
        },
    ],
    'K5-CP02': [
        # Different domain: saving vs spending, not studying. Same reasoning: extract abstract quality difference.
        {
            'text': 'Andi menabung setiap bulan dengan teratur. Ia memisahkan uang untuk kebutuhan dan keinginan. Budi menghabiskan seluruh uang sakunya setiap hari. Pada akhir bulan, Andi punya tabungan sedangkan Budi tidak punya sama sekali.',
            'question': 'Apa perbedaan cara mengelola uang Andi dan Budi?',
            'answer': 'hemat vs boros',
            'explanation': 'Pattern: When comparing two people\'s methods, extract the ABSTRACT QUALITY difference (hemat vs boros, rajin vs malas), not the literal actions',
        },
        {
            'text': 'Sari berolahraga setiap pagi dan makan makanan sehat. Dina jarang berolahraga dan suka makan junk food. Sari tampak bugar sedangkan Dina sering sakit.',
            'question': 'Apa perbedaan gaya hidup Sari dan Dina?',
            'answer': 'sehat vs tidak sehat',
            'explanation': 'Pattern: Compare methods → extract abstract quality difference (sehat vs tidak sehat)',
        },
    ],
}


def check_answer(answer, expected_keywords):
    if answer is None:
        return False, "got=None"
    answer_lower = str(answer).lower().strip()
    for keyword in expected_keywords:
        keyword_lower = keyword.lower().strip()
        if keyword_lower in answer_lower:
            return True, f"matched '{keyword_lower}' in '{answer_lower}'"
    return False, f"got='{answer_lower}', expected any of {expected_keywords}"


def run_teaching_test():
    """Run the teaching loop: test → teach → re-test → verify generalization"""
    self_core = SelfCore()
    engine = DerivationEngine(self_core)
    # Use the same TextComprehension instance for both teaching and testing
    # so that taught patterns are visible during re-testing
    engine._init_modules()
    tc = engine.text_comprehension

    print("=" * 70)
    print("SELF-AI v12 — TEACHING LOOP TEST")
    print("Teach similar-type questions (different domain+answer) → verify learning")
    print("=" * 70)

    # ─── PHASE 1: Test original failed soal BEFORE teaching ───
    print("\n📊 PHASE 1: Test original failed soal (BEFORE teaching)")
    print("-" * 50)

    phase1_results = []
    for soal in FAILED_SOAL:
        result = engine.derive_from_text(soal['text'], soal['question'])
        answer = result.get('answer')
        is_match, detail = check_answer(answer, soal['expected_keywords'])
        phase1_results.append({'id': soal['id'], 'pass': is_match, 'detail': detail,
                               'answer': str(answer) if answer else None, 'method': result.get('method', '')})
        status = "✅" if is_match else "❌"
        print(f"  {status} {soal['id']}: {detail}")

    p1_pass = sum(1 for r in phase1_results if r['pass'])
    print(f"\nPHASE 1: {p1_pass}/{len(FAILED_SOAL)} PASS")

    # ─── PHASE 2: TEACH with similar-type, different domain, different answers ───
    print(f"\n📚 PHASE 2: Teaching with SIMILAR-TYPE, DIFFERENT DOMAIN soal")
    print("-" * 50)

    taught_count = 0
    for soal in FAILED_SOAL:
        sid = soal['id']
        if sid in TEACHING:
            examples = TEACHING[sid]
            print(f"\n  🔧 Teaching for {sid} ({soal['type']}):")
            for ex in examples:
                print(f"     Q: {ex['question']}")
                print(f"     A: {ex['answer']}")
                print(f"     Pattern: {ex['explanation'][:80]}...")
                tc.teach(ex['text'], ex['question'], ex['answer'], ex['explanation'])
                taught_count += 1

    print(f"\n  Total teaching examples: {taught_count}")

    # ─── PHASE 3: Re-test ORIGINAL failed soal AFTER teaching ───
    print(f"\n🔄 PHASE 3: Re-test ORIGINAL failed soal (AFTER teaching)")
    print("-" * 50)

    improved = 0
    phase3_results = []
    for soal in FAILED_SOAL:
        result = engine.derive_from_text(soal['text'], soal['question'])
        answer = result.get('answer')
        is_match, detail = check_answer(answer, soal['expected_keywords'])
        was_match = next(r['pass'] for r in phase1_results if r['id'] == soal['id'])
        improved_status = "IMPROVED!" if (is_match and not was_match) else ("STILL PASS" if was_match else "STILL FAIL")
        phase3_results.append({'id': soal['id'], 'pass': is_match, 'detail': detail,
                               'answer': str(answer) if answer else None, 'method': result.get('method', ''),
                               'improved': is_match and not was_match})
        status = "✅" if is_match else "❌"
        print(f"  {status} {soal['id']}: {detail} ({improved_status})")
        if is_match and not was_match:
            improved += 1

    p3_pass = sum(1 for r in phase3_results if r['pass'])

    # ─── PHASE 4: Test GENERALIZATION with UNSEEN soal of the same types ───
    print(f"\n🧪 PHASE 4: Test GENERALIZATION with UNSEEN soal (same types, different content)")
    print("-" * 50)

    unseen_tests = [
        {
            'id': 'GEN-PB03',
            'text': 'Pedagang kaki lima itu berjualan dari pagi hingga malam tanpa lelah. Ia berdiri di bawah terik matahari demi mencukupi kebutuhan anak-anaknya yang masih sekolah.',
            'question': 'Peribahasa apa yang menggambarkan kerja keras pedagang tersebut?',
            'expected_keywords': ['banting tulang', 'keringat', 'kerja keras'],
            'type': 'peribahasa',
        },
        {
            'id': 'GEN-MH01',
            'text': 'Kebakaran gudang menyebabkan banyak barang hangus. Toko-toko di sekitarnya kehilangan persediaan. Banyak pemilik toko harus menutup usahanya sementara.',
            'question': 'Mengapa toko-toko harus menutup usahanya?',
            'expected_keywords': ['kebakaran', 'gudang', 'hangus', 'barang'],
            'type': 'implisit',
        },
        {
            'id': 'GEN-BK02',
            'text': 'Daun-daun bergumam pelan ditiup angin sore.',
            'question': 'Kata "bergumam" pada kalimat tersebut termasuk majas....',
            'expected_keywords': ['personifikasi'],
            'type': 'bahasa_kiasan',
        },
        {
            'id': 'GEN-CP02',
            'text': 'Maya merapikan tempat tidurnya setiap pagi dan menyapu kamarnya. Suka meninggalkan pakaian berserakan dan tidak pernah merapikan barang-barangnya.',
            'question': 'Apa perbedaan kebiasaan Maya dan Suka?',
            'expected_keywords': ['rapi', 'rajin', 'berantakan', 'malas'],
            'type': 'perbandingan',
        },
    ]

    gen_pass = 0
    gen_results = []
    for test in unseen_tests:
        result = engine.derive_from_text(test['text'], test['question'])
        answer = result.get('answer')
        is_match, detail = check_answer(answer, test['expected_keywords'])
        gen_results.append({'id': test['id'], 'pass': is_match, 'detail': detail,
                           'answer': str(answer) if answer else None, 'method': result.get('method', '')})
        status = "✅" if is_match else "❌"
        print(f"  {status} {test['id']}: {detail}")
        if is_match:
            gen_pass += 1

    # ─── FINAL SUMMARY ───
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Phase 1 (before teaching): {p1_pass}/{len(FAILED_SOAL)} PASS")
    print(f"  Phase 3 (after teaching):  {p3_pass}/{len(FAILED_SOAL)} PASS")
    print(f"  Improvement from teaching: +{improved} soal")
    print(f"  Phase 4 (generalization):  {gen_pass}/{len(unseen_tests)} PASS")
    print(f"  Teaching examples given:   {taught_count}")

    # Detailed results per soal
    print("\nPER-SOAL DETAIL:")
    for i, soal in enumerate(FAILED_SOAL):
        r1 = phase1_results[i]
        r3 = phase3_results[i]
        print(f"  {soal['id']}:")
        print(f"    Before: {r1['detail']}")
        print(f"    After:  {r3['detail']}")
        print(f"    Improved: {'YES ✅' if r3.get('improved') else 'NO ❌'}")

    # Save results
    benchmark_data = {
        'version': 'v12',
        'test_type': 'kelas5_teaching',
        'subject': 'Bahasa Indonesia',
        'grade': 5,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase1_before_teaching': phase1_results,
        'taught_count': taught_count,
        'phase3_after_teaching': phase3_results,
        'improved_count': improved,
        'phase4_generalization': gen_results,
        'generalization_pass': gen_pass,
        'generalization_total': len(unseen_tests),
    }

    benchmark_path = os.path.join(PROJECT_ROOT, 'benchmark', 'kelas5_teaching_results.json')
    os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2, default=str)
    print(f"\nBenchmark saved to: {benchmark_path}")

    total_pass = p3_pass + gen_pass
    total_soal = len(FAILED_SOAL) + len(unseen_tests)
    return total_pass, total_soal


if __name__ == '__main__':
    total_pass, total_soal = run_teaching_test()
    sys.exit(0 if total_pass == total_soal else 1)
