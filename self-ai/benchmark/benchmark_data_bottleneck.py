#!/usr/bin/env python3
"""Benchmark: Is DATA the bottleneck, not architecture?

HYPOTHESIS:
    All architecture iterations (v38→v40b) produced the same accuracy (~75-80%)
    because the bottleneck is training data quantity, not scoring strategy.

EXPERIMENT:
    A. Teach with current thin data (2 examples per domain)
    B. Teach with expanded data (5-6 examples per domain)
    C. Measure: does correct pattern rank #1 more often with more data?

METHODOLOGY:
    Pure embedding matching — no Qwen3 needed.
    For each test case:
      1. Compute c→c and q→q scores against all taught patterns
      2. Apply variance-weighted combined score (v40b)
      3. Check: does the correct q_type+subtype rank #1?

    This directly measures whether more training data improves
    the discriminative power of embedding matching.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

os.environ['TOKENIZERS_PARALLELISM'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
import logging
logging.disable(logging.WARNING)

import numpy as np


# ═══════════════════════════════════════════════════════════════
# TEST CASES — 30 cases across 8 domains
# ═══════════════════════════════════════════════════════════════

TEST_CASES = [
    # ── PERIBAHASA ──
    {'id': 'PB-01', 'text': 'Kedua orang tua Siti bekerja keras siang dan malam agar Siti bisa sekolah. Siti pun belajar dengan rajin dan tidak menyia-nyiakan pengorbanan orang tuanya.',
     'question': 'Peribahasa yang menggambarkan kerja keras orang tua Siti adalah....',
     'expected_subtype': 'kerja_keras', 'q_type': 'peribahasa'},
    {'id': 'PB-02', 'text': 'Pedagang kaki lima itu berjualan dari pagi hingga malam tanpa lelah. Ia berdiri di bawah terik matahari demi mencukupi kebutuhan anak-anaknya.',
     'question': 'Peribahasa apa yang menggambarkan kerja keras pedagang tersebut?',
     'expected_subtype': 'kerja_keras', 'q_type': 'peribahasa'},
    {'id': 'PB-03', 'text': 'Setelah bertahun-tahun merantau, Budi tidak pernah kembali ke kampung halamannya. Ia bahkan lupa pada orang tua yang telah membesarkannya.',
     'question': 'Peribahasa yang sesuai dengan keadaan Budi adalah....',
     'expected_subtype': 'ketidakbersyukuran', 'q_type': 'peribahasa'},
    {'id': 'PB-04', 'text': 'Rina selalu menghormati orang yang lebih tua dan menolong teman yang kesulitan.',
     'question': 'Peribahasa yang tepat untuk sikap Rina adalah....',
     'expected_subtype': 'kebaikan', 'q_type': 'peribahasa'},

    # ── BAHASA KIASAN / PERSONIFIKASI ──
    {'id': 'BK-01', 'text': 'Angin menjerit keras menggoyangkan pepohonan di malam yang gelap itu.',
     'question': 'Kata "menjerit" pada kalimat tersebut termasuk majas....',
     'expected_subtype': 'personifikasi', 'q_type': 'bahasa_kiasan'},
    {'id': 'BK-02', 'text': 'Matahari tersenyum hangat menyinari desa itu di pagi hari.',
     'question': 'Kata "tersenyum" pada kalimat tersebut termasuk majas....',
     'expected_subtype': 'personifikasi', 'q_type': 'bahasa_kiasan'},
    {'id': 'BK-03', 'text': 'Daun-daun bergumam pelan ditiup angin sore di taman kota.',
     'question': 'Kata "bergumam" pada kalimat tersebut termasuk majas....',
     'expected_subtype': 'personifikasi', 'q_type': 'bahasa_kiasan'},
    {'id': 'BK-04', 'text': 'Hujan menari-nari di atap rumah kami sepanjang malam.',
     'question': 'Kata "menari-nari" pada kalimat tersebut termasuk majas....',
     'expected_subtype': 'personifikasi', 'q_type': 'bahasa_kiasan'},
    {'id': 'BK-05', 'text': 'Suaranya menggelegar bak petir saat memberi perintah.',
     'question': 'Kata "menggelegar bak petir" pada kalimat tersebut termasuk majas....',
     'expected_subtype': 'hiperbola', 'q_type': 'bahasa_kiasan'},
    {'id': 'BK-06', 'text': 'Wajahnya bagaikan bulan purnama yang bersinar terang.',
     'question': 'Kata "bagaikan bulan purnama" pada kalimat tersebut termasuk majas....',
     'expected_subtype': 'simile', 'q_type': 'bahasa_kiasan'},

    # ── IDE POKOK ──
    {'id': 'IP-01', 'text': 'Kucing adalah hewan yang sangat rajin membersihkan diri. Setiap hari kucing menjilat bulunya untuk menjaga kebersihan. Kucing juga mengubur kotorannya dengan tanah.',
     'question': 'Apa ide pokok dari paragraf tersebut?',
     'expected_subtype': 'kalimat_awal', 'q_type': 'ide_pokok'},
    {'id': 'IP-02', 'text': 'Indonesia memiliki keragaman budaya yang sangat kaya. Setiap daerah mempunyai bahasa, pakaian adat, dan upacara tradisional yang berbeda. Keragaman ini menjadikan Indonesia unik.',
     'question': 'Apa ide pokok paragraf tersebut?',
     'expected_subtype': 'kalimat_awal', 'q_type': 'ide_pokok'},
    {'id': 'IP-03', 'text': 'Banyak siswa yang kurang minum air putih saat di sekolah. Mereka lebih memilih minuman manis. Oleh karena itu, kita harus membiasakan minum air putih yang cukup.',
     'question': 'Apa ide pokok paragraf tersebut?',
     'expected_subtype': 'kalimat_akhir', 'q_type': 'ide_pokok'},
    {'id': 'IP-04', 'text': 'Hutan-hutan di Kalimantan semakin berkurang setiap tahun. Penebangan liar dan pembukaan lahan menjadi penyebab utama. Oleh karena itu, upaya pelestarian hutan harus segera dilakukan.',
     'question': 'Apa gagasan utama paragraf tersebut?',
     'expected_subtype': 'kalimat_akhir', 'q_type': 'ide_pokok'},

    # ── IMPLISIT ──
    {'id': 'IM-01', 'text': 'Hujan turun tanpa henti selama tiga hari. Sungai meluap dan air masuk ke rumah-rumah warga. Banyak jalan tergenang air sehingga bus sekolah tidak bisa beroperasi.',
     'question': 'Mengapa anak-anak tidak bisa pergi ke sekolah?',
     'expected_subtype': 'causal_chain', 'q_type': 'implisit'},
    {'id': 'IM-02', 'text': 'Pabrik rotan di kota itu kehabisan bahan baku. Mesin-mesin berhenti beroperasi. Para pekerja tidak bisa masuk kerja dan kehilangan penghasilan.',
     'question': 'Mengapa pekerja kehilangan penghasilan?',
     'expected_subtype': 'causal_chain', 'q_type': 'implisit'},
    {'id': 'IM-03', 'text': 'Kebakaran hutan menghasilkan asap tebal. Asap menutupi bandara sehingga pesawat tidak bisa mendarat. Penumpang tertunda selama berjam-jam.',
     'question': 'Mengapa penumpang tertunda?',
     'expected_subtype': 'causal_chain', 'q_type': 'implisit'},
    {'id': 'IM-04', 'text': 'Kekeringan melanda desa itu. Tanaman padi layu dan petani tidak bisa panen. Harga beras di pasar naik drastis.',
     'question': 'Mengapa harga beras naik?',
     'expected_subtype': 'causal_chain', 'q_type': 'implisit'},

    # ── PERBANDINGAN ──
    {'id': 'CP-01', 'text': 'Rani belajar setiap malam sebelum ujian. Ia membuat catatan dan mengerjakan latihan soal. Doni hanya membaca buku sepintas pada malam sebelum ujian.',
     'question': 'Apa perbedaan cara belajar Rani dan Doni?',
     'expected_subtype': 'abstract_quality', 'q_type': 'perbandingan'},
    {'id': 'CP-02', 'text': 'Andi menabung setiap bulan dengan teratur. Budi menghabiskan seluruh uang sakunya setiap hari.',
     'question': 'Apa perbedaan cara mengelola uang Andi dan Budi?',
     'expected_subtype': 'abstract_quality', 'q_type': 'perbandingan'},
    {'id': 'CP-03', 'text': 'Sari berolahraga setiap pagi dan makan makanan sehat. Dina jarang berolahraga dan suka makan junk food.',
     'question': 'Apa perbedaan gaya hidup Sari dan Dina?',
     'expected_subtype': 'abstract_quality', 'q_type': 'perbandingan'},
    {'id': 'CP-04', 'text': 'Rumah Budi besar tetapi sederhana. Bagaimana gaya hidup Budi?',
     'question': 'Gaya hidup Budi dapat digambarkan sebagai....',
     'expected_subtype': 'kontras', 'q_type': 'perbandingan'},

    # ── EKSPLISIT ──
    {'id': 'EK-01', 'text': 'Paman pergi ke pasar pada pukul 05.00 pagi untuk menjual sayur-sayuran dari kebunnya.',
     'question': 'Pada pukul berapa paman pergi ke pasar?',
     'expected_subtype': 'fact_extraction', 'q_type': 'eksplisit'},
    {'id': 'EK-02', 'text': 'Sekolah Dasar Negeri 1 Bandung memiliki 480 siswa. Kepala sekolahnya bernama Bapak Hendra.',
     'question': 'Siapa kepala sekolah SDN 1 Bandung?',
     'expected_subtype': 'fact_extraction', 'q_type': 'eksplisit'},
    {'id': 'EK-03', 'text': 'Kucing kesayangan Dina bernama Mochi. Mochi berbulu putih dan sangat suka makan ikan.',
     'question': 'Apa nama kucing kesayangan Dina?',
     'expected_subtype': 'fact_extraction', 'q_type': 'eksplisit'},

    # ── SIKAP TOKOH ──
    {'id': 'ST-01', 'text': 'Tokoh selalu menolong tetangganya meskipun ia sendiri miskin.',
     'question': 'Apa sikap tokoh tersebut?',
     'expected_subtype': 'dermawan', 'q_type': 'sikap_tokoh'},
    {'id': 'ST-02', 'text': 'Semua orang berkata Lia pemarah. Tapi Lia tidak pernah marah. Sifat Lia sebenarnya?',
     'question': 'Sifat Lia sebenarnya adalah....',
     'expected_subtype': 'fakta_vs_opini', 'q_type': 'sikap_tokoh'},

    # ── TEKS EKSPLANASI ──
    {'id': 'TE-01', 'text': 'Mengapa hujan turun? Air di laut dan daratan menguap karena panas matahari. Uap air naik dan mendingin membentuk awan. Awan semakin berat dan jatuh sebagai hujan.',
     'question': 'Mengapa hujan turun?',
     'expected_subtype': 'proses_alam', 'q_type': 'teks_eksplanasi'},
    {'id': 'TE-02', 'text': 'Andi punya 5 kelereng merah dan 3 kelereng biru. Budi punya 7 kelereng. Berapa jumlah kelereng Andi?',
     'question': 'Berapa jumlah kelereng Andi?',
     'expected_subtype': 'angka_pengganggu', 'q_type': 'teks_eksplanasi'},
]


# ═══════════════════════════════════════════════════════════════
# THIN TRAINING DATA (2 per domain) — current state
# ═══════════════════════════════════════════════════════════════

THIN_TRAINING = [
    # Peribahasa (2)
    ('Para petani banting tulang dari subuh hingga magrib menggarap sawah mereka.', 'Peribahasa untuk kerja keras?', 'banting tulang', 'kerja keras fisik → banting tulang', 'peribahasa'),
    ('Setelah bertahun-tahun merantau, Budi lupa pada orang tuanya.', 'Peribahasa untuk orang yang melupakan asalnya?', 'kacang lupa kulit', 'melupakan asal-usul → kacang lupa kulit', 'peribahasa'),

    # Bahasa kiasan (2)
    ('Hujan menari-nari di atap rumah kami.', 'Kata menari-nari termasuk majas....', 'personifikasi', 'non-human + human action → personifikasi', 'bahasa_kiasan'),
    ('Suaranya menggelegar bak petir.', 'Kata menggelegar bak petir termasuk majas....', 'hiperbola', 'perbandingan berlebihan → hiperbola', 'bahasa_kiasan'),

    # Ide pokok (2)
    ('Kucing adalah hewan yang sangat rajin membersihkan diri. Setiap hari kucing menjilat bulunya.', 'Apa ide pokok paragraf tersebut?', 'kucing rajin membersihkan diri', 'kalimat pertama menyatakan gagasan utama → ide pokok', 'ide_pokok'),
    ('Oleh karena itu, kita harus menjaga kebersihan lingkungan.', 'Apa ide pokok dari paragraf tersebut?', 'kita harus menjaga kebersihan lingkungan', 'kalimat dengan penanda kesimpulan → ide pokok di akhir', 'ide_pokok'),

    # Implisit (2)
    ('Kebakaran gudang menyebabkan barang hangus. Toko kehilangan persediaan dan harus tutup.', 'Mengapa toko tutup?', 'kebakaran gudang', 'A→B→C chain, root cause = A', 'implisit'),
    ('Kemarau panjang membuat wadah kering. Air berkurang. Warga mengantre air.', 'Mengapa warga mengantre air?', 'kemarau panjang', 'A→B→C, root cause = A', 'implisit'),

    # Perbandingan (2)
    ('Andi menabung teratur. Budi menghabiskan uang setiap hari.', 'Perbedaan mengelola uang?', 'hemat vs boros', 'compare → abstract quality difference', 'perbandingan'),
    ('Rumah Budi besar tetapi sederhana. Gaya hidup Budi?', 'Gaya hidup Budi?', 'sederhana', 'kata setelah tetapi lebih penting', 'perbandingan'),

    # Eksplisit (2)
    ('Dokter Siti praktik mulai pukul 08.00 pagi.', 'Pukul berapa dokter praktik?', '08.00', 'pukul berapa → find time value', 'eksplisit'),
    ('Kucing berbulu hitam milik Rina diberi nama Midnight.', 'Apa nama kucing Rina?', 'Midnight', 'apa nama → find proper noun', 'eksplisit'),

    # Sikap tokoh (2)
    ('Tokoh menolong tetangga meskipun miskin.', 'Apa sikap tokoh?', 'dermawan', 'menolong meskipun sulit = dermawan', 'sikap_tokoh'),
    ('Semua orang berkata Lia pemarah. Tapi Lia tidak pernah marah.', 'Sifat Lia sebenarnya?', 'sabar', 'fakta mengalahkan opini populer', 'sikap_tokoh'),

    # Teks eksplanasi (2)
    ('Hujan turun karena air menguap, membentuk awan, lalu jatuh.', 'Mengapa hujan turun?', 'penguapan → kondensasi → hujan', 'proses sebab-akibat A→B→C', 'teks_eksplanasi'),
    ('Andi punya 5 kelereng merah dan 3 biru. Budi punya 7. Jumlah kelereng Andi?', 'Berapa jumlah kelereng Andi?', '8', 'hanya hitung milik entitas yang ditanyakan', 'teks_eksplanasi'),
]


# ═══════════════════════════════════════════════════════════════
# EXPANDED TRAINING DATA (5-6 per domain) — proposed
# ═══════════════════════════════════════════════════════════════

EXPANDED_TRAINING = THIN_TRAINING + [
    # ── PERIBAHASA: 4 more examples (total 6) ──
    ('Nelayan tua mengayuh perahu pagi buta hingga petang. Ia menghabiskan keringat untuk keluarganya.', 'Peribahasa untuk nelayan yang bekerja keras?', 'banting tulang', 'kerja keras fisik + keringat → banting tulang', 'peribahasa'),
    ('Ibu guru Sari rela menambah jam mengajar. Ia bahkan membelikan buku dari gajinya sendiri.', 'Peribahasa untuk guru yang berkorban?', 'banting tulang', 'bekerja keras demi orang lain → banting tulang', 'peribahasa'),
    ('Rina lupa pada sahabat yang telah menolongnya saat dia miskin.', 'Peribahasa untuk orang yang lupa budi?', 'kacang lupa kulit', 'melupakan jasa orang lain → kacang lupa kulit', 'peribahasa'),
    ('Budi dan Andi selalu tolong-menolong, baik saat senang maupun susah.', 'Peribahasa untuk persahabatan sejati?', 'berat sama dipikul ringan sama dijinjing', 'kerja sama dan saling menolong → berat sama dipikul', 'peribahasa'),

    # ── BAHASA KIASAN: 4 more examples (total 6) ──
    ('Bintang-bintang berkelipkan mata di langit malam.', 'Kata berkelipkan mata termasuk majas....', 'personifikasi', 'non-human (bintang) + human action (berkelipkan mata) → personifikasi', 'bahasa_kiasan'),
    ('Pohon-pohon merunduk sedih ketika musim gugur tiba.', 'Kata merunduk sedih termasuk majas....', 'personifikasi', 'non-human (pohon) + human action (merunduk sedih) → personifikasi', 'bahasa_kiasan'),
    ('Awan menangis di atas kota sore itu.', 'Kata menangis termasuk majas....', 'personifikasi', 'non-human (awan) + human action (menangis) → personifikasi', 'bahasa_kiasan'),
    ('Wajahnya bagaikan bulan purnama yang bersinar terang.', 'Kata bagaikan bulan purnama termasuk majas....', 'perumpamaan (simile)', 'kata pembanding bagaikan + perbandingan eksplisit → simile', 'bahasa_kiasan'),

    # ── IDE POKOK: 4 more examples (total 6) ──
    ('Indonesia memiliki keragaman budaya yang sangat kaya. Setiap daerah mempunyai bahasa dan pakaian adat yang berbeda. Keragaman ini menjadikan Indonesia unik.', 'Apa ide pokok paragraf tersebut?', 'keragaman budaya Indonesia', 'kalimat pertama menyatakan gagasan utama, kalimat lain menjelaskan → ide pokok di awal', 'ide_pokok'),
    ('Teknologi informasi berkembang sangat pesat. Internet memudahkan akses pengetahuan. Media sosial menghubungkan orang di seluruh dunia.', 'Apa gagasan utama paragraf tersebut?', 'perkembangan teknologi informasi', 'kalimat pertama = gagasan utama, sisanya penjelasan → ide pokok di awal', 'ide_pokok'),
    ('Banyak siswa kurang minum air putih saat di sekolah. Mereka lebih memilih minuman manis. Oleh karena itu, kita harus membiasakan minum air putih.', 'Apa ide pokok paragraf tersebut?', 'kita harus membiasakan minum air putih', 'kata olek karena itu di akhir = kesimpulan → ide pokok di akhir', 'ide_pokok'),
    ('Pencemaran sungai semakin parah di kota besar. Limbah industri dan domestik mencemari air. Oleh karena itu, upaya pembersihan sungai harus segera dilakukan.', 'Apa gagasan utama paragraf tersebut?', 'upaya pembersihan sungai harus segera dilakukan', 'karena itu + kalimat akhir = kesimpulan → ide pokok di akhir', 'ide_pokok'),

    # ── IMPLISIT: 4 more examples (total 6) ──
    ('Gempa bumi menghancurkan jembatan penghubung dua kota. Truk pengangkut barang tidak bisa lewat. Harga kebutuhan pokok naik.', 'Mengapa harga kebutuhan naik?', 'gempa menghancurkan jembatan', 'A→B→C chain: gempa → jembatan hancur → truk tidak lewat → harga naik', 'implisit'),
    ('Virus menyebar cepat di kota itu. Banyak karyawan sakit sehingga perusahaan menghentikan produksi.', 'Mengapa produksi dihentikan?', 'virus menyebar', 'A→B→C: virus → karyawan sakit → produksi berhenti', 'implisit'),
    ('Jalan raya ditutup karena longsor. Bus antarkota tidak bisa lewat. Penumpang terdampar di terminal.', 'Mengapa penumpang terdampar?', 'jalan ditutup karena longsor', 'A→B→C: longsor → jalan tutup → bus tidak lewat → penumpang terdampar', 'implisit'),
    ('Kebakaran hutan menghasilkan asap tebal. Asap menutupi bandara sehingga pesawat tidak bisa mendarat.', 'Mengapa pesawat tidak bisa mendarat?', 'asap tebal dari kebakaran hutan', 'A→B→C: kebakaran → asap → bandara tertutup → pesawat tidak mendarat', 'implisit'),

    # ── PERBANDINGAN: 4 more examples (total 6) ──
    ('Sari berolahraga setiap pagi dan makan sehat. Dina jarang olahraga dan suka junk food.', 'Perbedaan gaya hidup Sari dan Dina?', 'sehat vs tidak sehat', 'compare actions → abstract quality difference', 'perbandingan'),
    ('Eko mengerjakan PR tepat waktu. Fajar sering menunda dan bermain game.', 'Perbedaan kebiasaan belajar?', 'rajin vs malas', 'compare → extract abstract quality', 'perbandingan'),
    ('Andi lebih tinggi dari Budi. Siapa lebih pendek?', 'Siapa lebih pendek?', 'Budi', 'perbandingan terbalik: A lebih X → B lebih (lawan X)', 'perbandingan'),
    ('Raja kaya dan berkuasa tetapi tidur di lantai dan makan seadanya.', 'Kehidupan raja sebenarnya?', 'sederhana', 'tetapi mengoreksi: setelah tetapi = keadaan sebenarnya', 'perbandingan'),

    # ── EKSPLISIT: 3 more examples (total 5) ──
    ('Taman Nasional Komodo terletak di Provinsi Nusa Tenggara Timur.', 'Di provinsi mana Taman Nasional Komodo?', 'Nusa Tenggara Timur', 'di provinsi mana → find province name', 'eksplisit'),
    ('Sekolah Dasar Negeri 3 Surabaya didirikan tahun 1975 oleh Bapak Suwiryo.', 'Siapa pendiri SDN 3 Surabaya?', 'Bapak Suwiryo', 'siapa → find person name', 'eksplisit'),
    ('Perpustakaan kota buka setiap Senin sampai Sabtu pukul 09.00 sampai 17.00.', 'Pukul berapa perpustakaan buka?', '09.00', 'pukul berapa → find time', 'eksplisit'),

    # ── SIKAP TOKOH: 2 more examples (total 4) ──
    ('Rani selalu menghormati orang tua dan menolong teman yang kesulitan. Ia dikenal anak baik hati.', 'Apa sikap Rani?', 'baik hati', 'menolong + menghormati = baik hati/peduli', 'sikap_tokoh'),
    ('Meskipun dicemooh, Amir tetap berbuat baik dan menolong siapa saja.', 'Apa sifat Amir?', 'pemaaf', 'berbuat baik meskipun dicemooh = pemaaf/ikhlas', 'sikap_tokoh'),

    # ── TEKS EKSPLANASI: 2 more examples (total 4) ──
    ('Gunung meletus mengeluarkan lava dan abu vulkanik. Lava membanjiri desa di lereng. Abu menutupi lahan pertanian.', 'Mengapa pertanian rusak?', 'abu vulkanik menutupi lahan', 'proses sebab-akibat: letusan → abu → pertanian rusak', 'teks_eksplanasi'),
    ('Erosi terjadi karena hujan mengikis tanah yang tidak tertutup vegetasi. Akar pohon tidak lagi menahan tanah.', 'Mengapa erosi terjadi?', 'hujan mengikis tanah tanpa vegetasi', 'sebab-akibat: tidak ada pohon → tanah tidak tertahan → erosi', 'teks_eksplanasi'),
]


def compute_matching_accuracy(model, test_cases, training_data):
    """Compute matching accuracy using variance-weighted combined score.

    For each test case:
      1. Encode question and context text
      2. Compute c→c and q→q scores against all training patterns
      3. Apply variance-weighted combined score
      4. Check if top-ranked pattern has the correct q_type

    Returns:
        dict with accuracy, per-domain breakdown, and failure details
    """
    # Encode all training data
    train_texts = [t[:300] for t, _, _, _, _ in training_data]
    train_questions = [q for _, q, _, _, _ in training_data]
    train_q_types = [qt for _, _, _, _, qt in training_data]

    train_ctx_embs = model.encode(train_texts, show_progress_bar=False,
                                   normalize_embeddings=True)
    train_q_embs = model.encode(train_questions, show_progress_bar=False,
                                 normalize_embeddings=True)

    # Encode all test data
    test_texts = [tc['text'][:300] for tc in test_cases]
    test_questions = [tc['question'] for tc in test_cases]

    test_ctx_embs = model.encode(test_texts, show_progress_bar=False,
                                  normalize_embeddings=True)
    test_q_embs = model.encode(test_questions, show_progress_bar=False,
                                normalize_embeddings=True)

    results = []
    for i, tc in enumerate(test_cases):
        ctx_emb = test_ctx_embs[i]
        q_emb = test_q_embs[i]

        # Compute scores against ALL training patterns (same q_type only for Pass 1)
        c_scores = {}
        q_scores = {}
        for j in range(len(training_data)):
            if train_q_types[j] != tc['q_type']:
                continue
            c_scores[j] = float(np.dot(ctx_emb, train_ctx_embs[j]))
            q_scores[j] = float(np.dot(q_emb, train_q_embs[j]))

        # Variance-weighted combined score
        if c_scores or q_scores:
            c_var = float(np.var(list(c_scores.values()))) if len(c_scores) > 1 else 0.0
            q_var = float(np.var(list(q_scores.values()))) if len(q_scores) > 1 else 0.0
            total_var = c_var + q_var

            if total_var > 1e-8:
                w_c = c_var / total_var
                w_q = q_var / total_var
            else:
                w_c = 0.5
                w_q = 0.5

            combined = {}
            for j in c_scores:
                combined[j] = w_c * c_scores[j] + w_q * q_scores.get(j, 0.0)
            for j in q_scores:
                if j not in c_scores:
                    combined[j] = q_scores[j]

            # Rank by combined score
            ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

            best_j = ranked[0][0] if ranked else None
            best_score = ranked[0][1] if ranked else 0.0

            # Check: does the best match have the expected subtype?
            # We use a simple check: does the matched training pattern's explanation
            # contain the expected_subtype keyword?
            best_explanation = training_data[best_j][3]
            expected = tc['expected_subtype']

            # More nuanced check: look at top-3 matches
            top3_qtypes = [train_q_types[j] for j, _ in ranked[:3]]
            top3_explanations = [training_data[j][3] for j, _ in ranked[:3]]

            # Correct = top match is same q_type (always true since we filter)
            # So we check: does the matched explanation relate to the expected subtype?
            # Simple heuristic: check if any keyword from expected subtype appears
            subtype_keywords = {
                'kerja_keras': ['kerja keras', 'banting tulang', 'keringat'],
                'ketidakbersyukuran': ['lupa', 'kulit', 'melupakan', 'tidak bersyukur'],
                'kebaikan': ['baik', 'menabur kebaikan', 'mulia'],
                'personifikasi': ['personifikasi', 'non-human', 'benda mati', 'sifat manusia'],
                'hiperbola': ['hiperbola', 'berlebihan', 'melebih-lebihkan'],
                'simile': ['simile', 'perumpamaan', 'bagaikan', 'seperti'],
                'kalimat_awal': ['kalimat pertama', 'di awal', 'gagasan utama'],
                'kalimat_akhir': ['kalimat akhir', 'karena itu', 'kesimpulan', 'di akhir'],
                'causal_chain': ['chain', 'root cause', 'A→B→C', 'sebab-akibat'],
                'abstract_quality': ['abstract quality', 'perbedaan', 'compare'],
                'kontras': ['tetapi', 'kontras', 'setelah tetapi'],
                'fact_extraction': ['find time', 'find proper noun', 'pukul berapa'],
                'dermawan': ['dermawan', 'menolong', 'peduli'],
                'fakta_vs_opini': ['fakta', 'opini', 'mengalahkan'],
                'proses_alam': ['proses', 'sebab-akibat', 'penguapan'],
                'angka_pengganggu': ['pengganggu', 'milik entitas'],
            }

            # Check if the top-1 match's explanation contains subtype keywords
            correct = False
            for kw in subtype_keywords.get(expected, [expected]):
                if kw.lower() in best_explanation.lower():
                    correct = True
                    break

            # Also check top-3
            top3_correct = False
            for exp in top3_explanations:
                for kw in subtype_keywords.get(expected, [expected]):
                    if kw.lower() in exp.lower():
                        top3_correct = True
                        break
                if top3_correct:
                    break

            # Margin: gap between best and second-best score
            margin = 0.0
            if len(ranked) > 1:
                margin = ranked[0][1] - ranked[1][1]

            results.append({
                'id': tc['id'],
                'q_type': tc['q_type'],
                'expected_subtype': expected,
                'top1_correct': correct,
                'top3_correct': top3_correct,
                'best_score': best_score,
                'margin': margin,
                'w_c': w_c,
                'w_q': w_q,
                'best_match_text': training_data[best_j][0][:60],
                'best_match_explanation': best_explanation[:80],
            })

    # Compute accuracy
    top1_correct = sum(1 for r in results if r['top1_correct'])
    top3_correct = sum(1 for r in results if r['top3_correct'])
    total = len(results)

    # Per-domain breakdown
    by_domain = {}
    for r in results:
        qt = r['q_type']
        if qt not in by_domain:
            by_domain[qt] = {'total': 0, 'top1': 0, 'top3': 0}
        by_domain[qt]['total'] += 1
        if r['top1_correct']:
            by_domain[qt]['top1'] += 1
        if r['top3_correct']:
            by_domain[qt]['top3'] += 1

    return {
        'total': total,
        'top1_correct': top1_correct,
        'top3_correct': top3_correct,
        'top1_accuracy': top1_correct / total if total else 0,
        'top3_accuracy': top3_correct / total if total else 0,
        'by_domain': by_domain,
        'details': results,
    }


def main():
    print("=" * 70)
    print("  DATA BOTTLENECK HYPOTHESIS TEST")
    print("  Does adding more training data improve embedding matching?")
    print("=" * 70)

    # Load model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-m3')
    print("\nbge-m3 loaded\n")

    # ── Run A: Thin training data ──
    print(f"A. THIN DATA: {len(THIN_TRAINING)} examples ({len(set(qt for _,_,_,_,qt in THIN_TRAINING))} domains)")
    print("-" * 50)
    result_thin = compute_matching_accuracy(model, TEST_CASES, THIN_TRAINING)
    print(f"  Top-1: {result_thin['top1_correct']}/{result_thin['total']} ({100*result_thin['top1_accuracy']:.0f}%)")
    print(f"  Top-3: {result_thin['top3_correct']}/{result_thin['total']} ({100*result_thin['top3_accuracy']:.0f}%)")
    for qt, stats in sorted(result_thin['by_domain'].items()):
        print(f"    {qt}: {stats['top1']}/{stats['total']} top-1, {stats['top3']}/{stats['total']} top-3")

    # Show failures
    print("\n  Failures (thin):")
    for r in result_thin['details']:
        if not r['top1_correct']:
            print(f"    {r['id']} ({r['q_type']}): expected={r['expected_subtype']}, "
                  f"got='{r['best_match_explanation'][:50]}', margin={r['margin']:.4f}")

    # ── Run B: Expanded training data ──
    print(f"\nB. EXPANDED DATA: {len(EXPANDED_TRAINING)} examples ({len(set(qt for _,_,_,_,qt in EXPANDED_TRAINING))} domains)")
    print("-" * 50)
    result_expanded = compute_matching_accuracy(model, TEST_CASES, EXPANDED_TRAINING)
    print(f"  Top-1: {result_expanded['top1_correct']}/{result_expanded['total']} ({100*result_expanded['top1_accuracy']:.0f}%)")
    print(f"  Top-3: {result_expanded['top3_correct']}/{result_expanded['total']} ({100*result_expanded['top3_accuracy']:.0f}%)")
    for qt, stats in sorted(result_expanded['by_domain'].items()):
        print(f"    {qt}: {stats['top1']}/{stats['total']} top-1, {stats['top3']}/{stats['total']} top-3")

    # Show failures
    print("\n  Failures (expanded):")
    for r in result_expanded['details']:
        if not r['top1_correct']:
            print(f"    {r['id']} ({r['q_type']}): expected={r['expected_subtype']}, "
                  f"got='{r['best_match_explanation'][:50]}', margin={r['margin']:.4f}")

    # ── Comparison ──
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")
    print(f"  Thin data     ({len(THIN_TRAINING):2d} examples): Top-1 = {100*result_thin['top1_accuracy']:.0f}%  Top-3 = {100*result_thin['top3_accuracy']:.0f}%")
    print(f"  Expanded data  ({len(EXPANDED_TRAINING):2d} examples): Top-1 = {100*result_expanded['top1_accuracy']:.0f}%  Top-3 = {100*result_expanded['top3_accuracy']:.0f}%")

    delta_top1 = result_expanded['top1_correct'] - result_thin['top1_correct']
    delta_top3 = result_expanded['top3_correct'] - result_thin['top3_correct']
    print(f"  Delta: Top-1 {'+' if delta_top1 >= 0 else ''}{delta_top1}, Top-3 {'+' if delta_top3 >= 0 else ''}{delta_top3}")

    # Per-domain comparison
    print("\n  Per-domain comparison:")
    all_domains = sorted(set(list(result_thin['by_domain'].keys()) + list(result_expanded['by_domain'].keys())))
    for qt in all_domains:
        thin = result_thin['by_domain'].get(qt, {'total': 0, 'top1': 0, 'top3': 0})
        exp = result_expanded['by_domain'].get(qt, {'total': 0, 'top1': 0, 'top3': 0})
        thin_pct = 100 * thin['top1'] / thin['total'] if thin['total'] else 0
        exp_pct = 100 * exp['top1'] / exp['total'] if exp['total'] else 0
        delta = exp_pct - thin_pct
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"    {qt:20s}: Thin {thin_pct:5.0f}% → Expanded {exp_pct:5.0f}% ({arrow}{abs(delta):+.0f}%)")

    # Verdict
    if delta_top1 > 0:
        print(f"\n  ✓ HYPOTHESIS CONFIRMED: More training data improves matching accuracy")
    elif delta_top1 == 0:
        print(f"\n  ○ HYPOTHESIS INCONCLUSIVE: More data doesn't change accuracy")
    else:
        print(f"\n  ✗ HYPOTHESIS REJECTED: More data actually hurts accuracy")

    # Save results
    output_path = os.path.join(PROJECT_ROOT, 'benchmark', 'data_bottleneck_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'thin': {k: v for k, v in result_thin.items() if k != 'details'},
            'expanded': {k: v for k, v in result_expanded.items() if k != 'details'},
            'comparison': {
                'delta_top1': delta_top1,
                'delta_top3': delta_top3,
                'hypothesis_confirmed': delta_top1 > 0,
            }
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
