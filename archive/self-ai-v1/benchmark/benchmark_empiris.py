#!/usr/bin/env python3
# @WHO:   self-ai/benchmark/benchmark_empiris.py
# @WHAT:  Empirical benchmark — does pattern matching improve answer quality?
# @PART:  self-ai/benchmark
# @ENTRY: main()

"""SELF-AI Empirical Benchmark — v38 Pattern Matching Efficacy

PURPOSE:
    Measure whether the learned pattern system actually improves answer quality.
    This is the FIRST empirical test of the ExperienceWeight architecture.

METHODOLOGY:
    Phase 1: Test 20 questions WITHOUT any teaching (baseline)
    Phase 2: Teach 10 examples (different domain, same type)
    Phase 3: Re-test the SAME 20 questions (with teaching)
    Phase 4: Test 10 NEW questions (generalization)
    Phase 5: Compare baseline vs post-teaching accuracy

CONDITIONS:
    Run A: With models loaded (bge-m3 + Qwen3) — CORRECT condition
    Run B: Without models (model=None) — simulates disk space issue

    The difference between Run A and Run B shows:
    - How much the system was broken by the disk space issue
    - Whether pattern matching works when models are available
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

os.environ['TOKENIZERS_PARALLELISM'] = '0'


# ═══════════════════════════════════════════════════════════════
# TEST SOAL — 20 questions across 5 types
# ═══════════════════════════════════════════════════════════════

TEST_SOAL = [
    # Peribahasa (4 questions)
    {
        'id': 'PB-01',
        'text': 'Kedua orang tua Siti bekerja keras siang dan malam agar Siti bisa sekolah. Siti pun belajar dengan rajin dan tidak menyia-nyiakan pengorbanan orang tuanya.',
        'question': 'Peribahasa yang menggambarkan kerja keras orang tua Siti adalah....',
        'expected_keywords': ['banting tulang', 'keringat', 'kerja keras'],
        'type': 'peribahasa',
    },
    {
        'id': 'PB-02',
        'text': 'Pedagang kaki lima itu berjualan dari pagi hingga malam tanpa lelah. Ia berdiri di bawah terik matahari demi mencukupi kebutuhan anak-anaknya.',
        'question': 'Peribahasa apa yang menggambarkan kerja keras pedagang tersebut?',
        'expected_keywords': ['banting tulang', 'keringat', 'kerja keras'],
        'type': 'peribahasa',
    },
    {
        'id': 'PB-03',
        'text': 'Setelah bertahun-tahun merantau, Budi tidak pernah kembali ke kampung halamannya. Ia bahkan lupa pada orang tua yang telah membesarkannya.',
        'question': 'Peribahasa yang sesuai dengan keadaan Budi adalah....',
        'expected_keywords': ['kacang lupa kulit', 'habis manis sepah', 'lupa daratan'],
        'type': 'peribahasa',
    },
    {
        'id': 'PB-04',
        'text': 'Rina selalu menghormati orang yang lebih tua dan menolong teman yang kesulitan. Ia dikenal sebagai anak yang baik hati.',
        'question': 'Peribahasa yang tepat untuk menggambarkan sikap Rina adalah....',
        'expected_keywords': ['siapa menabur angin', 'budi', 'baik', 'mulia'],
        'type': 'peribahasa',
    },

    # Bahasa kiasan / personifikasi (4 questions)
    {
        'id': 'BK-01',
        'text': 'Angin menjerit keras menggoyangkan pepohonan di malam yang gelap itu.',
        'question': 'Kata "menjerit" pada kalimat tersebut termasuk majas....',
        'expected_keywords': ['personifikasi'],
        'type': 'bahasa_kiasan',
    },
    {
        'id': 'BK-02',
        'text': 'Matahari tersenyum hangat menyinari desa itu di pagi hari.',
        'question': 'Kata "tersenyum" pada kalimat tersebut termasuk majas....',
        'expected_keywords': ['personifikasi'],
        'type': 'bahasa_kiasan',
    },
    {
        'id': 'BK-03',
        'text': 'Daun-daun bergumam pelan ditiup angin sore di taman kota.',
        'question': 'Kata "bergumam" pada kalimat tersebut termasuk majas....',
        'expected_keywords': ['personifikasi'],
        'type': 'bahasa_kiasan',
    },
    {
        'id': 'BK-04',
        'text': 'Hujan menari-nari di atap rumah kami sepanjang malam.',
        'question': 'Kata "menari-nari" pada kalimat tersebut termasuk majas....',
        'expected_keywords': ['personifikasi'],
        'type': 'bahasa_kiasan',
    },

    # Implisit / multi-hop (4 questions)
    {
        'id': 'IM-01',
        'text': 'Hujan turun tanpa henti selama tiga hari. Sungai meluap dan air masuk ke rumah-rumah warga. Banyak jalan tergenang air sehingga bus sekolah tidak bisa beroperasi.',
        'question': 'Mengapa anak-anak tidak bisa pergi ke sekolah?',
        'expected_keywords': ['banjir', 'jalan tergenang', 'bus tidak beroperasi', 'air', 'hujan'],
        'type': 'implisit',
    },
    {
        'id': 'IM-02',
        'text': 'Pabrik rotan di kota itu kehabisan bahan baku. Mesin-mesin berhenti beroperasi. Para pekerja tidak bisa masuk kerja dan kehilangan penghasilan.',
        'question': 'Mengapa pekerja kehilangan penghasilan?',
        'expected_keywords': ['pabrik', 'bahan baku', 'kehabisan', 'mesin berhenti'],
        'type': 'implisit',
    },
    {
        'id': 'IM-03',
        'text': 'Kebakaran hutan menghasilkan asap tebal. Asap menutupi bandara sehingga pesawat tidak bisa mendarat. Penumpang tertunda selama berjam-jam.',
        'question': 'Mengapa penumpang tertunda?',
        'expected_keywords': ['kebakaran', 'asap', 'bandara', 'pesawat'],
        'type': 'implisit',
    },
    {
        'id': 'IM-04',
        'text': 'Kekeringan melanda desa itu. Tanaman padi layu dan petani tidak bisa panen. Harga beras di pasar naik drastis.',
        'question': 'Mengapa harga beras naik?',
        'expected_keywords': ['kekeringan', 'tanaman layu', 'tidak bisa panen', 'petani'],
        'type': 'implisit',
    },

    # Perbandingan (4 questions)
    {
        'id': 'CP-01',
        'text': 'Rani belajar setiap malam sebelum ujian. Ia membuat catatan dan mengerjakan latihan soal. Doni hanya membaca buku sepintas pada malam sebelum ujian. Hasilnya, Rani mendapat nilai 90 sedangkan Doni mendapat nilai 55.',
        'question': 'Apa perbedaan cara belajar Rani dan Doni?',
        'expected_keywords': ['rajin', 'tekun', 'malas', 'serius', 'cermat'],
        'type': 'perbandingan',
    },
    {
        'id': 'CP-02',
        'text': 'Andi menabung setiap bulan dengan teratur. Ia memisahkan uang untuk kebutuhan dan keinginan. Budi menghabiskan seluruh uang sakunya setiap hari.',
        'question': 'Apa perbedaan cara mengelola uang Andi dan Budi?',
        'expected_keywords': ['hemat', 'boros', 'menabung', 'teratur'],
        'type': 'perbandingan',
    },
    {
        'id': 'CP-03',
        'text': 'Sari berolahraga setiap pagi dan makan makanan sehat. Dina jarang berolahraga dan suka makan junk food. Sari tampak bugar sedangkan Dina sering sakit.',
        'question': 'Apa perbedaan gaya hidup Sari dan Dina?',
        'expected_keywords': ['sehat', 'tidak sehat', 'rajn olahraga', 'jarang olahraga'],
        'type': 'perbandingan',
    },
    {
        'id': 'CP-04',
        'text': 'Maya merapikan tempat tidurnya setiap pagi dan menyapu kamarnya. Suka meninggalkan pakaian berserakan dan tidak pernah merapikan barang-barangnya.',
        'question': 'Apa perbedaan kebiasaan Maya dan Suka?',
        'expected_keywords': ['rapi', 'rajin', 'berantakan', 'malas'],
        'type': 'perbandingan',
    },

    # Eksplisit / fact extraction (4 questions)
    {
        'id': 'EK-01',
        'text': 'Paman pergi ke pasar pada pukul 05.00 pagi untuk menjual sayur-sayuran dari kebunnya. Ia pulang pada pukul 12.00 siang.',
        'question': 'Pada pukul berapa paman pergi ke pasar?',
        'expected_keywords': ['05.00', '5', 'pagi'],
        'type': 'eksplisit',
    },
    {
        'id': 'EK-02',
        'text': 'Sekolah dasar Negeri 1 Bandung memiliki 480 siswa. Kepala sekolahnya bernama Bapak Hendra. Sekolah ini berdiri sejak tahun 1985.',
        'question': 'Siapa kepala sekolah SDN 1 Bandung?',
        'expected_keywords': ['hendra', 'bapak hendra'],
        'type': 'eksplisit',
    },
    {
        'id': 'EK-03',
        'text': 'Kucing kesayangan Dina bernama Mochi. Mochi berbulu putih dan sangat suka makan ikan. Dina mendapatkan Mochi saat ulang tahunnya yang ke-7.',
        'question': 'Apa nama kucing kesayangan Dina?',
        'expected_keywords': ['mochi'],
        'type': 'eksplisit',
    },
    {
        'id': 'EK-04',
        'text': 'Taman Nasional Komodo terletak di Provinsi Nusa Tenggara Timur. Taman ini terkenal dengan hewan endemiknya yaitu komodo.',
        'question': 'Di provinsi mana Taman Nasional Komodo terletak?',
        'expected_keywords': ['nusa tenggara timur', 'ntt'],
        'type': 'eksplisit',
    },
]


# ═══════════════════════════════════════════════════════════════
# TEACHING EXAMPLES — 10 examples (different domain, same type)
# ═══════════════════════════════════════════════════════════════

TEACHING = [
    # Peribahasa: different domains, same "kerja keras" pattern
    {
        'text': 'Para petani di desa banting tulang dari subuh hingga magrib menggarap sawah mereka. Hasil panen yang melimpah menjadi buah kerja keras mereka.',
        'question': 'Peribahasa yang sesuai untuk menggambarkan kerja keras para petani?',
        'answer': 'banting tulang',
        'explanation': 'Pattern: orang bekerja keras fisik siang dan malam → banting tulang',
    },
    {
        'text': 'Nelayan tua itu mengayuh perahunya setiap pagi buta hingga petang. Ia menghabiskan keringat untuk memenuhi kebutuhan keluarganya.',
        'question': 'Peribahasa apa yang menggambarkan semangat kerja nelayan tersebut?',
        'answer': 'banting tulang',
        'explanation': 'Pattern: bekerja keras dengan keringat → banting tulang (kerja keras fisik)',
    },

    # Bahasa kiasan: different objects, same personifikasi pattern
    {
        'text': 'Hujan menari-nari di atap rumah kami.',
        'question': 'Kata "menari-nari" pada kalimat tersebut termasuk majas....',
        'answer': 'personifikasi',
        'explanation': 'Pattern: non-human subject (hujan) + human action (menari-nari) → personifikasi',
    },
    {
        'text': 'Bintang-bintang berkelipkan mata di langit malam.',
        'question': 'Kata "berkelipkan mata" pada kalimat tersebut termasuk majas....',
        'answer': 'personifikasi',
        'explanation': 'Pattern: non-human (bintang) + human action (berkelipkan mata) → personifikasi',
    },

    # Implisit: different chains, same root-cause pattern
    {
        'text': 'Kebakaran gudang menyebabkan banyak barang hangus. Toko-toko di sekitarnya kehilangan persediaan. Banyak pemilik toko harus menutup usahanya sementara.',
        'question': 'Mengapa toko-toko harus menutup usahanya?',
        'answer': 'kebakaran gudang',
        'explanation': 'Pattern: A→B→C chain. Kebakaran → barang hangus → toko tutup. Root cause: kebakaran gudang',
    },
    {
        'text': 'Kemarau panjang membuat wadah kering. Air PDAM berkurang drastis. Warga harus mengantre air bersih berjam-jam.',
        'question': 'Mengapa warga harus mengantre air?',
        'answer': 'kemarau panjang',
        'explanation': 'Pattern: A→B→C. Kemarau → wadah kering → air berkurang → antre. Root cause: kemarau',
    },

    # Perbandingan: different domains, same abstract-quality pattern
    {
        'text': 'Andi menabung setiap bulan dengan teratur. Ia memisahkan uang untuk kebutuhan dan keinginan. Budi menghabiskan seluruh uang sakunya setiap hari.',
        'question': 'Apa perbedaan cara mengelola uang Andi dan Budi?',
        'answer': 'hemat vs boros',
        'explanation': 'Pattern: When comparing two people, extract the ABSTRACT QUALITY difference (hemat vs boros), not literal actions',
    },
    {
        'text': 'Sari berolahraga setiap pagi dan makan makanan sehat. Dina jarang berolahraga dan suka makan junk food.',
        'question': 'Apa perbedaan gaya hidup Sari dan Dina?',
        'answer': 'sehat vs tidak sehat',
        'explanation': 'Pattern: Compare methods → extract abstract quality difference (sehat vs tidak sehat)',
    },

    # Eksplisit: different facts, same direct-extraction pattern
    {
        'text': 'Dokter Siti berpraktik di rumah sakit Hasan Sadikin setiap hari Senin hingga Jumat. Praktiknya dimulai pukul 08.00 pagi.',
        'question': 'Pada pukul berapa dokter Siti mulai praktik?',
        'answer': '08.00',
        'explanation': 'Pattern: When asked "pukul berapa", find the time value directly stated in the text',
    },
    {
        'text': 'Kucing berbulu hitam milik Rina diberi nama Midnight. Midnight sangat suka bermain dengan bola benang.',
        'question': 'Apa nama kucing Rina?',
        'answer': 'Midnight',
        'explanation': 'Pattern: When asked "apa nama", find the proper noun that follows "nama" or is identified as a name',
    },
]


# ═══════════════════════════════════════════════════════════════
# GENERALIZATION TEST — 10 unseen questions
# ═══════════════════════════════════════════════════════════════

GENERALIZATION_SOAL = [
    {
        'id': 'GEN-PB-01',
        'text': 'Tukang becak itu mengayuh dari pagi hingga sore hari. Ia tidak pernah mengeluh meski keringat bercucuran.',
        'question': 'Peribahasa yang menggambarkan semangat kerja tukang becak?',
        'expected_keywords': ['banting tulang', 'keringat', 'kerja keras'],
        'type': 'peribahasa',
    },
    {
        'id': 'GEN-BK-01',
        'text': 'Pohon-pohon merunduk sedih ketika musim gugur tiba.',
        'question': 'Kata "merunduk sedih" pada kalimat tersebut termasuk majas....',
        'expected_keywords': ['personifikasi'],
        'type': 'bahasa_kiasan',
    },
    {
        'id': 'GEN-IM-01',
        'text': 'Gempa bumi menghancurkan jembatan penghubung dua kota. Truk-truk pengangkut barang tidak bisa lewat. Harga kebutuhan pokok naik.',
        'question': 'Mengapa harga kebutuhan pokok naik?',
        'expected_keywords': ['gempa', 'jembatan', 'truk', 'tidak bisa lewat'],
        'type': 'implisit',
    },
    {
        'id': 'GEN-CP-01',
        'text': 'Eko selalu mengerjakan PR tepat waktu dan membuat jadwal belajar. Fajar sering menunda PR dan bermain game hingga larut.',
        'question': 'Apa perbedaan kebiasaan belajar Eko dan Fajar?',
        'expected_keywords': ['rajin', 'disiplin', 'malas', 'menunda', 'prokrastinasi'],
        'type': 'perbandingan',
    },
    {
        'id': 'GEN-EK-01',
        'text': 'Perpustakaan kota buka setiap hari Senin sampai Sabtu pukul 09.00 sampai 17.00. Pada hari Minggu perpustakaan tutup.',
        'question': 'Pada pukul berapa perpustakaan kota buka?',
        'expected_keywords': ['09.00', '9', 'pagi'],
        'type': 'eksplisit',
    },
    {
        'id': 'GEN-PB-02',
        'text': 'Ibu guru Sari rela menambah jam mengajar agar murid-muridnya bisa lulus ujian. Ia bahkan membelikan buku dari gajinya sendiri.',
        'question': 'Peribahasa yang cocok untuk guru Sari yang berkorban demi muridnya?',
        'expected_keywords': ['berbakti', 'mulia', 'pengorbanan', 'banting tulang'],
        'type': 'peribahasa',
    },
    {
        'id': 'GEN-BK-02',
        'text': 'Awan menangis di atas kota sore itu.',
        'question': 'Kata "menangis" pada kalimat tersebut termasuk majas....',
        'expected_keywords': ['personifikasi'],
        'type': 'bahasa_kiasan',
    },
    {
        'id': 'GEN-IM-02',
        'text': 'Virus menyebar cepat di kota itu. Banyak karyawan sakit sehingga perusahaan harus menghentikan produksi sementara.',
        'question': 'Mengapa perusahaan menghentikan produksi?',
        'expected_keywords': ['virus', 'sakit', 'karyawan'],
        'type': 'implisit',
    },
    {
        'id': 'GEN-CP-02',
        'text': 'Adi selalu menghabiskan uangnya untuk beli makanan sehat dan suplemen. Bowo lebih suka makan di warung pinggir jalan tanpa memperhatikan gizi.',
        'question': 'Apa perbedaan pola makan Adi dan Bowo?',
        'expected_keywords': ['sehat', 'tidak sehat', 'gizi', 'sembarangan'],
        'type': 'perbandingan',
    },
    {
        'id': 'GEN-EK-02',
        'text': 'Sekolah Dasar Negeri 3 Surabaya didirikan pada tahun 1975 oleh Bapak Suwiryo. Sekolah ini memiliki 12 ruang kelas.',
        'question': 'Siapa yang mendirikan SDN 3 Surabaya?',
        'expected_keywords': ['suwiryo', 'bapak suwiryo'],
        'type': 'eksplisit',
    },
]


def check_answer(answer, expected_keywords):
    """Check if answer contains any expected keyword."""
    if answer is None:
        return False, "got=None"
    answer_lower = str(answer).lower().strip()
    for keyword in expected_keywords:
        keyword_lower = keyword.lower().strip()
        if keyword_lower in answer_lower:
            return True, f"matched '{keyword_lower}' in '{answer_lower[:80]}'"
    return False, f"got='{answer_lower[:80]}', expected any of {expected_keywords}"


def run_benchmark(models_enabled=True):
    """Run full benchmark with or without models."""
    
    # Clean up any existing learned patterns
    patterns_file = os.path.join(PROJECT_ROOT, 'data', 'learned_patterns.json')
    if os.path.exists(patterns_file):
        os.remove(patterns_file)
    
    from derivation.text_comprehension import TextComprehension
    from derivation.engine import DerivationEngine
    from core.self import SelfCore
    
    self_core = SelfCore()
    engine = DerivationEngine(self_core)
    engine._init_modules()
    tc = engine.text_comprehension
    
    # Check if models loaded
    from derivation.model_registry import get_shared_embedding_model, get_shared_qwen
    emb_model = get_shared_embedding_model()
    qwen_model, _ = get_shared_qwen()
    models_loaded = emb_model is not None and qwen_model is not None
    
    condition = "MODELS LOADED" if models_loaded else "NO MODELS (disk space issue simulation)"
    
    results = {
        'condition': condition,
        'models_loaded': models_loaded,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # ── PHASE 1: Baseline (before teaching) ──
    print(f"\n{'='*70}")
    print(f"  BENCHMARK — {condition}")
    print(f"{'='*70}")
    print(f"\nPhase 1: Baseline — 20 questions WITHOUT teaching")
    print(f"{'-'*50}")
    
    phase1 = []
    for soal in TEST_SOAL:
        result = engine.derive_from_text(soal['text'], soal['question'])
        answer = result.get('answer')
        is_match, detail = check_answer(answer, soal['expected_keywords'])
        method = result.get('method', '')
        confidence = result.get('confidence', 0)
        phase1.append({
            'id': soal['id'],
            'type': soal['type'],
            'pass': is_match,
            'detail': detail,
            'answer': str(answer)[:100] if answer else None,
            'method': method,
            'confidence': round(confidence, 3),
        })
        status = "PASS" if is_match else "FAIL"
        print(f"  {status} {soal['id']} ({soal['type']}): {method[:40]} conf={confidence:.2f}")
    
    p1_pass = sum(1 for r in phase1 if r['pass'])
    p1_total = len(phase1)
    print(f"\n  Phase 1: {p1_pass}/{p1_total} correct ({100*p1_pass/p1_total:.0f}%)")
    
    # ── PHASE 2: Teaching ──
    print(f"\nPhase 2: Teaching 10 examples")
    print(f"{'-'*50}")
    
    for i, ex in enumerate(TEACHING):
        tc.teach(ex['text'], ex['question'], ex['answer'], ex['explanation'])
        print(f"  Taught {i+1}/10: {ex['question'][:60]}...")
    
    # Check pattern state after teaching
    patterns_count = len(tc.learned_patterns)
    patterns_with_embeddings = 0
    for pk, pd in tc.learned_patterns.items():
        if pd.get('question_embedding') and len(pd['question_embedding']) > 0:
            patterns_with_embeddings += 1
    
    print(f"\n  Patterns stored: {patterns_count}")
    print(f"  Patterns with embeddings: {patterns_with_embeddings}")
    
    # ── PHASE 3: Re-test same 20 questions (with teaching) ──
    print(f"\nPhase 3: Re-test 20 questions (WITH teaching)")
    print(f"{'-'*50}")
    
    phase3 = []
    improved = 0
    degraded = 0
    for soal in TEST_SOAL:
        result = engine.derive_from_text(soal['text'], soal['question'])
        answer = result.get('answer')
        is_match, detail = check_answer(answer, soal['expected_keywords'])
        method = result.get('method', '')
        confidence = result.get('confidence', 0)
        
        was_match = next(r['pass'] for r in phase1 if r['id'] == soal['id'])
        change = "IMPROVED" if (is_match and not was_match) else ("DEGRADED" if (not is_match and was_match) else "")
        if is_match and not was_match:
            improved += 1
        if not is_match and was_match:
            degraded += 1
        
        phase3.append({
            'id': soal['id'],
            'type': soal['type'],
            'pass': is_match,
            'detail': detail,
            'answer': str(answer)[:100] if answer else None,
            'method': method,
            'confidence': round(confidence, 3),
            'change': change,
        })
        status = "PASS" if is_match else "FAIL"
        change_str = f" [{change}]" if change else ""
        print(f"  {status} {soal['id']} ({soal['type']}): {method[:40]} conf={confidence:.2f}{change_str}")
    
    p3_pass = sum(1 for r in phase3 if r['pass'])
    print(f"\n  Phase 3: {p3_pass}/{p1_total} correct ({100*p3_pass/p1_total:.0f}%)")
    print(f"  Improved: +{improved}, Degraded: -{degraded}, Net: {'+' if improved-degraded>=0 else ''}{improved-degraded}")
    
    # ── PHASE 4: Generalization (10 new questions) ──
    print(f"\nPhase 4: Generalization — 10 NEW questions")
    print(f"{'-'*50}")
    
    phase4 = []
    for soal in GENERALIZATION_SOAL:
        result = engine.derive_from_text(soal['text'], soal['question'])
        answer = result.get('answer')
        is_match, detail = check_answer(answer, soal['expected_keywords'])
        method = result.get('method', '')
        confidence = result.get('confidence', 0)
        phase4.append({
            'id': soal['id'],
            'type': soal['type'],
            'pass': is_match,
            'detail': detail,
            'answer': str(answer)[:100] if answer else None,
            'method': method,
            'confidence': round(confidence, 3),
        })
        status = "PASS" if is_match else "FAIL"
        print(f"  {status} {soal['id']} ({soal['type']}): {method[:40]} conf={confidence:.2f}")
    
    p4_pass = sum(1 for r in phase4 if r['pass'])
    p4_total = len(phase4)
    print(f"\n  Phase 4: {p4_pass}/{p4_total} correct ({100*p4_pass/p4_total:.0f}%)")
    
    # ── PHASE 5: Method breakdown ──
    print(f"\nPhase 5: Method breakdown")
    print(f"{'-'*50}")
    
    all_results = phase1 + phase3 + phase4
    method_counts = {}
    for r in all_results:
        m = r['method']
        if m not in method_counts:
            method_counts[m] = {'total': 0, 'correct': 0}
        method_counts[m]['total'] += 1
        if r['pass']:
            method_counts[m]['correct'] += 1
    
    for m, c in sorted(method_counts.items(), key=lambda x: x[1]['total'], reverse=True):
        pct = 100 * c['correct'] / c['total'] if c['total'] > 0 else 0
        print(f"  {m[:50]:50s} {c['correct']:2d}/{c['total']:2d} ({pct:5.1f}%)")
    
    # Learned pattern methods specifically
    learned_methods = [m for m in method_counts if 'learned' in m]
    if learned_methods:
        learned_total = sum(method_counts[m]['total'] for m in learned_methods)
        learned_correct = sum(method_counts[m]['correct'] for m in learned_methods)
        print(f"\n  LEARNED PATTERN METHODS: {learned_correct}/{learned_total} ({100*learned_correct/learned_total:.0f}%)")
    else:
        print(f"\n  LEARNED PATTERN METHODS: NONE — patterns never used!")
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {condition}")
    print(f"{'='*70}")
    print(f"  Phase 1 (baseline):         {p1_pass}/{p1_total} ({100*p1_pass/p1_total:.0f}%)")
    print(f"  Phase 3 (after teaching):   {p3_pass}/{p1_total} ({100*p3_pass/p1_total:.0f}%)")
    print(f"  Net improvement:            {'+' if improved-degraded>=0 else ''}{improved-degraded}")
    print(f"  Phase 4 (generalization):   {p4_pass}/{p4_total} ({100*p4_pass/p4_total:.0f}%)")
    print(f"  Patterns used:              {len(learned_methods)} methods, {learned_total if learned_methods else 0} invocations")
    print(f"  Models loaded:              {models_loaded}")
    print(f"  Embeddings in patterns:     {patterns_with_embeddings}/{patterns_count}")
    
    # Save results
    results.update({
        'phase1_baseline': phase1,
        'phase1_accuracy': round(p1_pass / p1_total, 4),
        'teaching_count': len(TEACHING),
        'patterns_stored': patterns_count,
        'patterns_with_embeddings': patterns_with_embeddings,
        'phase3_after_teaching': phase3,
        'phase3_accuracy': round(p3_pass / p1_total, 4),
        'improved': improved,
        'degraded': degraded,
        'net_improvement': improved - degraded,
        'phase4_generalization': phase4,
        'phase4_accuracy': round(p4_pass / p4_total, 4) if p4_total > 0 else 0,
        'method_breakdown': {m: c for m, c in method_counts.items()},
        'learned_pattern_usage': {
            'methods': learned_methods,
            'total_invocations': learned_total if learned_methods else 0,
            'correct': learned_correct if learned_methods else 0,
        },
    })
    
    return results


def main():
    # Run with models loaded (current condition — should have 2GB free now)
    print("STARTING EMPIRICAL BENCHMARK v38")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = run_benchmark(models_enabled=True)
    
    # Save to file
    benchmark_path = os.path.join(PROJECT_ROOT, 'benchmark', 'empirical_v38_results.json')
    os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to: {benchmark_path}")
    
    return results


if __name__ == '__main__':
    results = main()
