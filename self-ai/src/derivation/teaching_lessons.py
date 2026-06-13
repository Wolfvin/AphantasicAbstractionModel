# @WHO:   self-ai/src/derivation/teaching_lessons.py
# @WHAT:  Structured teaching lessons — soal + cara penyelesaian + jawaban + penjelasan kenapa
# @PART:  self-ai/derivation
# @ENTRY: TeachingLessons (imported by pattern_learner.py)

"""Teaching Lessons — the Teacher mechanism for SELF-AI.

Philosophy:
    Teacher provides:
        1. Soal (problem/question)
        2. Cara penyelesaian (solution steps / method)
        3. Jawaban (answer)
        4. Penjelasan kenapa (explanation of WHY this answer)

    SELF observes these examples and DISCOVERS its own semantic patterns
    through inner thinking. SELF writes its own patterns — NOT the teacher.

    The teacher does NOT give axioms. The teacher gives EXAMPLES with
    explanations. SELF must find the patterns.

    Like a math teacher:
        Soal: "3 + 5 = ?"
        Cara: "Tambahkan 3 dan 5"
        Jawaban: "8"
        Kenapa: "Karena penjumlahan menggabungkan dua bilangan menjadi satu"

    SELF then discovers: "penjumlahan = menggabungkan bilangan"
    This is SELF's own semantic pattern, not the teacher's axiom.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TeachingLesson:
    """A single structured teaching example.

    Attributes:
        problem: The question/soal being asked
        solution_steps: How to solve it (cara penyelesaian)
        answer: The correct answer (jawaban)
        explanation_why: WHY this is the answer (penjelasan kenapa)
        question_type: Category of question (e.g., 'peribahasa', 'ide_pokok')
        context_text: Optional text passage the question is about
        difficulty: Optional difficulty level (1-5)
    """

    def __init__(self, problem: str, solution_steps: list, answer: str,
                 explanation_why: str, question_type: str = '',
                 context_text: str = '', difficulty: int = 3):
        self.problem = problem
        self.solution_steps = solution_steps
        self.answer = answer
        self.explanation_why = explanation_why
        self.question_type = question_type
        self.context_text = context_text
        self.difficulty = difficulty

    def to_dict(self) -> dict:
        return {
            'problem': self.problem,
            'solution_steps': self.solution_steps,
            'answer': self.answer,
            'explanation_why': self.explanation_why,
            'question_type': self.question_type,
            'context_text': self.context_text,
            'difficulty': self.difficulty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TeachingLesson':
        return cls(
            problem=d.get('problem', ''),
            solution_steps=d.get('solution_steps', []),
            answer=d.get('answer', ''),
            explanation_why=d.get('explanation_why', ''),
            question_type=d.get('question_type', ''),
            context_text=d.get('context_text', ''),
            difficulty=d.get('difficulty', 3),
        )

    def __repr__(self):
        return f"TeachingLesson(type={self.question_type}, q={self.problem[:50]}...)"


class TeachingLessons:
    """Collection of structured teaching lessons for SELF-AI.

    These are EXAMPLES that SELF observes. SELF must discover
    its own patterns from these examples through inner thinking.

    The teacher does NOT tell SELF what patterns to find.
    The teacher provides examples and SELF figures out the patterns.
    """

    def __init__(self):
        self._lessons = []
        self._by_type = {}

    def add(self, lesson: TeachingLesson):
        """Add a teaching lesson."""
        self._lessons.append(lesson)
        qt = lesson.question_type
        if qt not in self._by_type:
            self._by_type[qt] = []
        self._by_type[qt].append(lesson)

    def get_by_type(self, question_type: str) -> list:
        """Get all lessons for a given question type."""
        return self._by_type.get(question_type, [])

    def get_all(self) -> list:
        """Get all lessons."""
        return list(self._lessons)

    def get_types(self) -> list:
        """Get all question types that have lessons."""
        return list(self._by_type.keys())

    def count(self) -> int:
        return len(self._lessons)

    def to_list(self) -> list:
        return [l.to_dict() for l in self._lessons]

    @classmethod
    def from_list(cls, items: list) -> 'TeachingLessons':
        tl = cls()
        for item in items:
            tl.add(TeachingLesson.from_dict(item))
        return tl


# ═══════════════ DEFAULT LESSONS ═══════════════
# These are starter examples. SELF will discover patterns from these.
# More lessons can be added dynamically via teach().

def get_default_lessons() -> TeachingLessons:
    """Get the default set of teaching lessons for SELF-AI.

    These are EXAMPLES for SELF to observe — NOT axioms for SELF to memorize.
    SELF must discover its own semantic patterns through inner thinking.
    """
    lessons = TeachingLessons()

    # ── PERIBAHASA ──
    lessons.add(TeachingLesson(
        problem="Apa peribahasa yang cocok untuk cerita tentang anak yang lupa pada orang tuanya yang bekerja keras?",
        solution_steps=[
            "Identifikasi situasi: seseorang yang lupa/melupakan",
            "Identifikasi konteks: orang tua bekerja keras untuk anak",
            "Cocokkan dengan peribahasa tentang ketidakbersyukuran",
        ],
        answer="kacang lupa akan kulitnya",
        explanation_why="Peribahasa ini berarti orang yang melupakan asal-usulnya atau tidak bersyukur atas jasa orang lain. Seperti kacang yang lupa bahwa ia tumbuh dari kulitnya.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Apa peribahasa untuk orang yang bekerja keras dan kemudian mendapat hasil?",
        solution_steps=[
            "Identifikasi pola: usaha keras dulu → hasil baik kemudian",
            "Cocokkan dengan peribahasa tentang usaha dan hasil",
        ],
        answer="bersakit-sakit dahulu bersenang-senang kemudian",
        explanation_why="Peribahasa ini berarti bahwa penderitaan atau kerja keras di awal akan menghasilkan kebahagiaan di kemudian hari. Pola: effort → reward.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Apa peribahasa untuk orang yang bekerja sama dan saling menolong?",
        solution_steps=[
            "Identifikasi situasi: kerja sama, tolong-menolong",
            "Cocokkan dengan peribahasa tentang kebersamaan",
        ],
        answer="berat sama dipikul ringan sama dijinjing",
        explanation_why="Peribahasa ini berarti dalam keadaan susah maupun senang, kita harus saling membantu. Pola: cooperation → mutual support.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Apa peribahasa untuk orang yang berbuat jahat dan mendapat balas jahat?",
        solution_steps=[
            "Identifikasi pola: perbuatan buruk → akibat buruk",
            "Cocokkan dengan peribahaaan tentang sebab-akibat",
        ],
        answer="siapa menabur angin akan menuai badai",
        explanation_why="Peribahaaan ini berarti siapa yang berbuat kejahatan akan menerima akibat dari kejahatannya. Pola: bad action → bad consequence.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Apa peribahasa untuk orang tua yang bekerja keras demi anaknya?",
        solution_steps=[
            "Identifikasi: orang tua bekerja sangat keras",
            "Cocokkan dengan peribahasa tentang kerja keras orang tua",
        ],
        answer="banting tulang",
        explanation_why="Banting tulang berarti bekerja keras dengan seluruh tenaga. Cocok untuk orang tua yang mengorbankan tenaga demi keluarga.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Apa peribahaaan untuk orang yang berbuat baik dan mendapat balasan baik?",
        solution_steps=[
            "Identifikasi pola: kebaikan → balasan kebaikan",
            "Cocokkan dengan peribahasa tentang kebaikan",
        ],
        answer="siapa menabur kebaikan akan menuai kebaikan",
        explanation_why="Peribahasa ini berarti kebaikan yang kita lakukan akan kembali kepada kita. Pola: kindness → kindness returned.",
        question_type='peribahasa',
    ))

    # ── BAHASA KIASAN ──
    lessons.add(TeachingLesson(
        problem="Apa gaya bahasa dalam kalimat 'angin menari-nari di antara pepohonan'?",
        solution_steps=[
            "Identifikasi subjek: angin (benda mati)",
            "Identifikasi kata kerja: menari-nari (sifat manusia)",
            "Kesimpulan: benda mati diberi sifat manusia = personifikasi",
        ],
        answer="personifikasi",
        explanation_why="Personifikasi adalah gaya bahasa yang memberikan sifat manusia pada benda mati. 'Menari-nari' adalah aktivitas manusia, diberikan pada 'angin' yang adalah benda mati.",
        question_type='bahasa_kiasan',
    ))

    lessons.add(TeachingLesson(
        problem="Apa gaya bahasa dalam kalimat 'wajahnya bagaikan bulan purnama'?",
        solution_steps=[
            "Identifikasi kata pembanding: bagaikan",
            "Identifikasi yang dibandingkan: wajahnya dengan bulan purnama",
            "Kesimpulan: perbandingan eksplisit = perumpamaan/simile",
        ],
        answer="perumpamaan (simile)",
        explanation_why="Simile adalah perbandingan eksplisit menggunakan kata pembanding seperti 'bagaikan', 'seperti', 'laksana'. Kata 'bagaikan' menandakan simile.",
        question_type='bahasa_kiasan',
    ))

    lessons.add(TeachingLesson(
        problem="Apa gaya bahasa dalam kalimat 'suaranya menggelegar bak petir'?",
        solution_steps=[
            "Identifikasi: bak petir = perbandingan berlebihan",
            "Menggelegar bak petir = melebih-lebihkan",
            "Kesimpulan: hiperbola",
        ],
        answer="hiperbola",
        explanation_why="Hiperbola adalah gaya bahasa yang melebih-lebihkan. 'Bak petir' melebih-lebihkan kekuatan suara — bukan perbandingan biasa karena skalanya mustahil.",
        question_type='bahasa_kiasan',
    ))

    # ── IDE POKOK ──
    lessons.add(TeachingLesson(
        problem="Apa ide pokok dari paragraf: 'Kucing adalah hewan yang sangat rajin membersihkan diri. Setiap hari kucing menjilat bulunya untuk menjaga kebersihan. Kucing juga mengubur kotorannya dengan tanah.'",
        solution_steps=[
            "Baca kalimat pertama: 'Kucing adalah hewan yang sangat rajin membersihkan diri'",
            "Kalimat pertama menyatakan gagasan utama tentang kebersihan kucing",
            "Kalimat berikutnya menjelaskan detail cara kucing membersihkan diri",
            "Kesimpulan: kalimat pertama = ide pokok",
        ],
        answer="kucing rajin membersihkan diri",
        explanation_why="Dalam teks eksposisi, ide pokok sering ada di kalimat pertama. Kalimat pertama menyatakan gagasan utama, kalimat berikutnya adalah penjelasan.",
        question_type='ide_pokok',
    ))

    lessons.add(TeachingLesson(
        problem="Apa ide pokok dari paragraf yang kalimat terakhirnya: 'Oleh karena itu, kita harus selalu menjaga kebersihan lingkungan.'",
        solution_steps=[
            "Identifikasi kata penanda kesimpulan: 'Oleh karena itu'",
            "Kalimat dengan penanda kesimpulan biasanya berisi ide pokok",
            "Kesimpulan: ide pokok ada di kalimat terakhir",
        ],
        answer="kita harus menjaga kebersihan lingkungan",
        explanation_why="Dalam teks argumentatif, ide pokok sering di akhir paragraf, ditandai oleh kata 'oleh karena itu', 'jadi', 'maka'. Ini disebut kalimat utama di akhir.",
        question_type='ide_pokok',
    ))

    # ── KECUALI / EXCEPTION ──
    lessons.add(TeachingLesson(
        problem="Semua siswa hadir kecuali Ani. Siapa yang tidak hadir?",
        solution_steps=[
            "Identifikasi kata kunci: 'kecuali' = pengecualian",
            "Artinya: semua hadir, TAPI Ani TIDAK hadir",
            "'Kecuali' membalikkan makna untuk yang disebut setelahnya",
        ],
        answer="Ani",
        explanation_why="Kata 'kecuali' menandakan pengecualian. Semua hadir, TAPI yang disebut setelah 'kecuali' TIDAK hadir. Jadi Ani tidak hadir. Pola: SEMUA A kecuali B → B TIDAK A.",
        question_type='pertanyaan_negatif',
    ))

    # ── KONTRAS / TETAPI ──
    lessons.add(TeachingLesson(
        problem="Rumah Budi besar tetapi sederhana. Bagaimana gaya hidup Budi?",
        solution_steps=[
            "Identifikasi kata 'tetapi' = penanda kontras",
            "'Besar tetapi sederhana' = meskipun besar, hidupnya sederhana",
            "Kontras menunjukkan bahwa 'sederhana' adalah sifat yang menonjol",
        ],
        answer="sederhana",
        explanation_why="Kata 'tetapi' menandakan kontras. Kata setelah 'tetapi' menunjukkan sifat yang lebih penting/dominan. 'Sederhana' adalah sifat yang menonjol dari Budi.",
        question_type='perbandingan',
    ))

    # ── PERBANDINGAN ──
    lessons.add(TeachingLesson(
        problem="Budi lebih tinggi dari Andi. Siapa yang lebih pendek?",
        solution_steps=[
            "Identifikasi: Budi > Andi (tinggi)",
            "Pertanyaan bertanya yang lebih pendek",
            "Jika Budi lebih tinggi, maka Andi lebih pendek",
        ],
        answer="Andi",
        explanation_why="Perbandingan 'lebih tinggi dari' membuat relasi terurut. Jika A > B dalam tinggi, maka B < A dalam tinggi, yang berarti B lebih pendek.",
        question_type='perbandingan',
    ))

    # ── BENAR/SALAH ──
    lessons.add(TeachingLesson(
        problem="Pernyataan: 'Jakarta terletak di Pulau Kalimantan.' Benar atau salah?",
        solution_steps=[
            "Cek pengetahuan: Jakarta adalah ibu kota Indonesia",
            "Jakarta terletak di Pulau Jawa, BUKAN Kalimantan",
            "Pernyataan bertentangan dengan fakta → salah",
        ],
        answer="salah",
        explanation_why="Jakarta terletak di Pulau Jawa. Pernyataan yang bertentangan dengan fakta adalah salah. Untuk menentukan benar/salah, kita harus membandingkan pernyataan dengan pengetahuan yang benar.",
        question_type='benar_salah',
    ))

    # ── TEKS ARGUMENTATIF ──
    lessons.add(TeachingLesson(
        problem="Dalam teks argumentatif, kalimat mana yang merupakan opini?",
        solution_steps=[
            "Cari kata-kata yang menunjukkan pendapat pribadi",
            "Penanda opini: 'menurut saya', 'sebaiknya', 'seharusnya'",
            "Penanda fakta: data, angka, penelitian",
            "Kalimat dengan penanda opini = kalimat opini",
        ],
        answer="Kalimat yang mengandung kata 'menurut saya', 'sebaiknya', atau 'seharusnya'",
        explanation_why="Opini adalah pendapat pribadi yang ditandai oleh kata-kata seperti 'menurut saya', 'sebaiknya'. Fakta adalah pernyataan yang bisa dibuktikan dengan data. Perbedaan kunci: opini = subjektif, fakta = objektif.",
        question_type='teks_argumentatif',
    ))

    # ── SIKAP TOKOH ──
    lessons.add(TeachingLesson(
        problem="Tokoh selalu menolong tetangganya meskipun ia sendiri miskin. Apa sikap tokoh tersebut?",
        solution_steps=[
            "Identifikasi tindakan: menolong tetangga",
            "Identifikasi konteks: meskipun miskin (berkorban)",
            "Menolong meskipun sulit = dermawan/peduli",
        ],
        answer="dermawan atau peduli",
        explanation_why="Sikap dermawan/peduli ditunjukkan melalui tindakan menolong orang lain meskipun dalam keadaan sulit. Kata kunci: menolong, berkorban, meskipun.",
        question_type='sikap_tokoh',
    ))

    # ── KESAN/PESAN ──
    lessons.add(TeachingLesson(
        problem="Cerita tentang anak yang rajin belajar dan akhirnya menjadi juara. Apa pesan cerita?",
        solution_steps=[
            "Identifikasi pola: rajin belajar → menjadi juara",
            "Pesan: kerja keras menghasilkan keberhasilan",
            "Ini adalah pesan moral tentang pentingnya kerja keras",
        ],
        answer="Kita harus rajin belajar agar berhasil",
        explanation_why="Pesan cerita adalah ajaran moral yang bisa diambil. Pola usaha → keberhasilan menunjukkan pesan bahwa kerja keras menghasilkan hasil yang baik.",
        question_type='kesan_pesan',
    ))

    # ── TEKS EKSPLANASI ──
    lessons.add(TeachingLesson(
        problem="Mengapa hujan turun?",
        solution_steps=[
            "Identifikasi pertanyaan sebab-akibat: 'mengapa'",
            "Cari proses: pemanasan air → penguapan → kondensasi → hujan",
            "Jelaskan rantai sebab-akibat",
        ],
        answer="Hujan turun karena air di laut dan daratan menguap, uap air naik dan mendingin membentuk awan, lalu jatuh sebagai hujan",
        explanation_why="Teks eksplanasi menjelaskan proses sebab-akibat. 'Mengapa' menanyakan penyebab. Jawaban harus menjelaskan rantai proses dari awal sampai akhir.",
        question_type='teks_eksplanasi',
    ))

    # ── ANALOGI ──
    lessons.add(TeachingLesson(
        problem="Dokter : Rumah Sakit = Guru : ?",
        solution_steps=[
            "Identifikasi relasi: dokter BEKERJA DI rumah sakit",
            "Terapkan relasi yang sama: guru BEKERJA DI ?",
            "Guru bekerja di sekolah",
        ],
        answer="sekolah",
        explanation_why="Analogi menggunakan relasi yang sama. Relasi 'bekerja di' menghubungkan dokter dengan rumah sakit. Relasi yang sama menghubungkan guru dengan sekolah.",
        question_type='analogi',
    ))

    # ── TONE/MOOD ──
    lessons.add(TeachingLesson(
        problem="Bagaimana suasana dalam cerita tentang anak yang kehilangan orang tuanya?",
        solution_steps=[
            "Identifikasi peristiwa: kehilangan orang tua",
            "Emosi yang muncul: sedih, duka, kehilangan",
            "Suasana = keseluruhan perasaan yang dominan",
        ],
        answer="sedih",
        explanation_why="Suasana (tone/mood) ditentukan oleh emosi dominan dalam cerita. Kehilangan orang tua menghasilkan suasana sedih/duka. Peristiwa tragis → suasana sedih.",
        question_type='tone_mood',
    ))

    # ── IMPLISIT ──
    lessons.add(TeachingLesson(
        problem="Andi selalu membawa payung meskipun cerah. Apa yang bisa disimpulkan?",
        solution_steps=[
            "Fakta eksplisit: Andi selalu bawa payung",
            "Pertanyaan: apa yang TIDAK langsung dikatakan tapi bisa disimpulkan?",
            "Orang bawa payung = antisipasi hujan",
            "Selalu bawa = sangat berhati-hati/antisipatif",
        ],
        answer="Andi adalah orang yang berhati-hati atau antisipatif",
        explanation_why="Implisit berarti tidak langsung dikatakan tapi bisa disimpulkan dari fakta. Kebiasaan selalu membawa payung menunjukkan sifat berhati-hati.",
        question_type='implisit',
    ))

    # ── SINONIM/ANTONIM ──
    lessons.add(TeachingLesson(
        problem="Apa lawan kata dari 'rajin'?",
        solution_steps=[
            "Identifikasi: lawan kata = antonim",
            "Rajin = suka bekerja keras",
            "Antonim = malas = tidak suka bekerja keras",
        ],
        answer="malas",
        explanation_why="Antonim adalah kata yang berlawanan makna. Rajin (suka bekerja) berlawanan dengan malas (tidak suka bekerja). Keduanya mendeskripsikan sikap terhadap kerja.",
        question_type='sinonim_antonim',
    ))

    # ═══════════════════════════════════════════════════════════════
    # ADVERSARIAL LESSONS — Guru mengajarkan pola-pola sulit
    # Ini adalah contoh-contoh yang mengajarkan SELF cara menangani
    # pertanyaan yang dirancang untuk MENIPU sistem.
    # ═══════════════════════════════════════════════════════════════

    # ── PENGECUALIAN (KECUALI) — Pola pengecualian membalikkan makna ──
    lessons.add(TeachingLesson(
        problem="Semua siswa hadir kecuali Ani. Siapa yang tidak hadir?",
        solution_steps=[
            "Temukan kata pengecualian: 'kecuali'",
            "Pisahkan: [semua siswa hadir] kecuali [Ani]",
            "Yang disebut setelah 'kecuali' TIDAK ikut aturan umum",
            "Aturan umum = hadir, jadi Ani TIDAK hadir",
        ],
        answer="Ani",
        explanation_why="Kata 'kecuali' membuat pengecualian. Aturan umum berlaku untuk semua KECUALI yang disebut setelahnya. Yang disebut setelah 'kecuali' memiliki keadaan BERLAWANAN. Pola: SEMUA A kecuali B → B TIDAK A.",
        question_type='pertanyaan_negatif',
        difficulty=4,
    ))

    lessons.add(TeachingLesson(
        problem="Semua hewan di kebun itu jinak selain harimau. Hewan mana yang bukan jinak?",
        solution_steps=[
            "Temukan kata pengecualian: 'selain'",
            "Pisahkan: [semua hewan jinak] selain [harimau]",
            "Harimau TIDAK jinak karena disebut setelah 'selain'",
        ],
        answer="harimau",
        explanation_why="'Selain' sama dengan 'kecuali' — membuat pengecualian. Aturan umum = jinak, pengecualian = harimau. Harimau berkeadaan berlawanan dari aturan umum.",
        question_type='pertanyaan_negatif',
        difficulty=4,
    ))

    # ── KONTRAS (TETAPI/NAMUN) — Kata setelah kontras lebih penting ──
    lessons.add(TeachingLesson(
        problem="Rumah Budi besar tetapi sederhana. Bagaimana gaya hidup Budi?",
        solution_steps=[
            "Temukan kata kontras: 'tetapi'",
            "Sebelum 'tetapi': besar (kesan pertama)",
            "Setelah 'tetapi': sederhana (kesan sebenarnya)",
            "Kata setelah 'tetapi' mengalahkan kata sebelumnya",
        ],
        answer="sederhana",
        explanation_why="Kata 'tetapi' atau 'namun' menandakan kontras. Apa yang dikatakan SETELAH 'tetapi' adalah keadaan sebenarnya yang lebih penting. Kata sebelumnya hanya kesan awal yang dikoreksi. Pola: A tetapi B → B lebih penting dari A.",
        question_type='perbandingan',
        difficulty=4,
    ))

    lessons.add(TeachingLesson(
        problem="Raja sangat kaya dan berkuasa tetapi dia tidur di lantai dan makan seadanya. Bagaimana kehidupan Raja sehari-hari?",
        solution_steps=[
            "Temukan kata kontras: 'tetapi'",
            "Sebelum tetapi: kaya dan berkuasa (kesan awal)",
            "Setelah tetapi: tidur di lantai dan makan seadanya (kenyataan)",
            "Kehidupan sehari-hari = yang SETELAH tetapi = sederhana",
        ],
        answer="sederhana",
        explanation_why="Meskipun raja kaya, kata 'tetapi' menunjukkan kehidupan sebenarnya bertolak belakang. Tidur di lantai dan makan seadanya = hidup sederhana. Kontras mengoreksi kesan awal.",
        question_type='perbandingan',
        difficulty=5,
    ))

    # ── NEGASI GANDA — "bukan/tidak" + kata negatif = positif ──
    lessons.add(TeachingLesson(
        problem="Siti bukan anak yang malas, tapi anak yang sangat rajin. Sifat Siti?",
        solution_steps=[
            "Temukan pola: 'bukan X, tapi Y'",
            "'bukan malas' = TIDAK malas",
            "'tapi rajin' = justru rajin",
            "Jawaban: sifat yang SETELAH 'tapi' = rajin",
        ],
        answer="rajin",
        explanation_why="Pola 'bukan X, tapi Y' menolak X dan menegaskan Y. 'Bukan malas' menolak sifat malas, 'tapi rajin' menegaskan sifat rajin. Jawaban selalu yang ditegaskan (setelah tapi).",
        question_type='pertanyaan_negatif',
        difficulty=4,
    ))

    # ── PERBANDINGAN TERBALIK — "lebih" + pertanyaan berlawanan ──
    lessons.add(TeachingLesson(
        problem="Andi lebih pendek dari Budi. Siapa yang lebih tinggi?",
        solution_steps=[
            "Identifikasi relasi: Andi < Budi (tinggi)",
            "Pertanyaan bertanya yang LEBIH TINGGI",
            "Jika Andi lebih pendek, maka Budi lebih tinggi",
            "Jawaban: pihak yang BERLAWANAN dari yang disebut",
        ],
        answer="Budi",
        explanation_why="Perbandingan 'lebih pendek dari' berarti Budi lebih tinggi. Pertanyaan membalik arah perbandingan. Jika A lebih X dari B, maka B lebih (lawan X) dari A.",
        question_type='perbandingan',
        difficulty=4,
    ))

    lessons.add(TeachingLesson(
        problem="Tinggi Andi 150 cm. Tinggi Budi 160 cm. Siapa yang lebih tinggi?",
        solution_steps=[
            "Bandingkan angka: 150 vs 160",
            "160 > 150, jadi Budi lebih tinggi",
            "Jawaban: Budi (angka lebih besar)",
        ],
        answer="Budi",
        explanation_why="Dengan angka konkret, bandingkan langsung. 160 > 150 → Budi lebih tinggi dari Andi. Jangan memilih yang disebut pertama — pilih berdasarkan perbandingan.",
        question_type='perbandingan',
        difficulty=4,
    ))

    # ── FAKTA vs OPINI POPULER — Teks mengoreksi opini ──
    lessons.add(TeachingLesson(
        problem="Semua orang berkata Lia pemarah. Tapi Lia tidak pernah marah. Sifat Lia sebenarnya?",
        solution_steps=[
            "Identifikasi dua sumber: 'semua orang berkata' vs 'Lia tidak pernah marah'",
            "'Semua orang berkata' = opini populer (bisa salah)",
            "'Lia tidak pernah marah' = fakta dari cerita",
            "Fakta mengalahkan opini populer",
        ],
        answer="sabar",
        explanation_why="Opini populer (semua orang berkata) BUKAN fakta. Fakta dalam teks (Lia tidak pernah marah) mengoreksi opini. Jawaban berdasarkan FAKTA dalam teks, bukan apa yang dikatakan orang.",
        question_type='sikap_tokoh',
        difficulty=5,
    ))

    # ── FAKTA dari TEKS EMOSIONAL — Pisahkan emosi dari fakta ──
    lessons.add(TeachingLesson(
        problem="Dengan hati berduka, ibu memasak nasi putih dan sayur. Apa yang ibu masak?",
        solution_steps=[
            "Pisahkan emosi dari fakta",
            "Emosi: berduka, sedih (bukan yang ditanyakan)",
            "Fakta: memasak nasi putih dan sayur",
            "Pertanyaan bertanya FAKTA, bukan emosi",
        ],
        answer="nasi putih dan sayur",
        explanation_why="Pertanyaan 'apa yang dimasak' bertanya fakta, bukan emosi. Meskipun teks penuh emosi (berduka), jawaban harus berdasarkan fakta (memasak nasi putih dan sayur). Jangan terpengaruh emosi dominan.",
        question_type='eksplisit',
        difficulty=4,
    ))

    # ── ANGKA PENGGANGGU — Hanya hitung yang ditanyakan ──
    lessons.add(TeachingLesson(
        problem="Andi punya 5 kelereng merah dan 3 kelereng biru. Budi punya 7 kelereng. Berapa jumlah kelereng Andi?",
        solution_steps=[
            "Identifikasi siapa yang ditanyakan: Andi",
            "Kelereng Andi: 5 merah + 3 biru = 8",
            "Kelereng Budi (7) TIDAK ditanyakan — angka pengganggu",
            "Jawaban: hanya hitung milik Andi = 8",
        ],
        answer="8",
        explanation_why="Pertanyaan hanya bertanya tentang Andi. Angka milik orang lain (Budi: 7) adalah pengganggu yang TIDAK dihitung. Hanya jumlahkan angka milik entitas yang ditanyakan.",
        question_type='teks_eksplanasi',
        difficulty=4,
    ))

    # ── PENCARIAN FAKTA SPESIFIK — Jawaban tersembunyi di satu kalimat ──
    lessons.add(TeachingLesson(
        problem="Jakarta adalah ibu kota Indonesia. Banyak gedung pencakar langit. Makanan khas Jakarta adalah kerak telor. Kota ini padat lalu lintas. Apa makanan khas Jakarta menurut teks?",
        solution_steps=[
            "Cari kata kunci: 'makanan khas'",
            "Temukan kalimat yang menyebut 'makanan khas': kerak telor",
            "Abaikan informasi lain (ibu kota, gedung, lalu lintas)",
        ],
        answer="kerak telor",
        explanation_why="Jawaban ada di satu kalimat spesifik. Jangan terpengaruh informasi lain dalam teks. Cari kata kunci pertanyaan ('makanan khas') di teks dan ambil jawabannya.",
        question_type='eksplisit',
        difficulty=4,
    ))

    # ═══════════════════════════════════════════════════════════════
    # EXPANDED LESSONS — More examples per subtype for better centroids
    # v41: Empirically verified that thin data (1-2 examples per subtype)
    # is the accuracy bottleneck, not architecture. Adding 3-4 more
    # examples per thin subtype improves embedding matching by +4-25%.
    # ═══════════════════════════════════════════════════════════════

    # ── PERIBAHASA: expanded from 6→10 ──
    lessons.add(TeachingLesson(
        problem="Nelayan tua mengayuh perahu pagi buta hingga petang. Ia menghabiskan keringat untuk keluarganya. Peribahasa untuk semangat kerjanya?",
        solution_steps=[
            "Identifikasi: bekerja keras secara fisik dari pagi hingga petang",
            "Keringat = kerja keras fisik yang luar biasa",
            "Cocokkan: kerja keras fisik → banting tulang",
        ],
        answer="banting tulang",
        explanation_why="Nelayan bekerja keras secara fisik sepanjang hari dengan menghabiskan keringat. Pola kerja keras fisik + keringat → banting tulang.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Ibu guru Sari rela menambah jam mengajar. Ia bahkan membelikan buku dari gajinya sendiri demi murid-muridnya. Peribahasa untuk pengorbanannya?",
        solution_steps=[
            "Identifikasi: bekerja keras demi orang lain, mengorbankan milik sendiri",
            "Kerja keras + pengorbanan → banting tulang",
            "Banting tulang juga berlaku untuk pengorbanan dalam bekerja keras",
        ],
        answer="banting tulang",
        explanation_why="Ibu guru bekerja keras dan mengorbankan gajinya sendiri demi murid. Kerja keras + pengorbanan → banting tulang. Pola: usaha keras fisik/mental demi orang lain.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Rina selalu menghormati orang yang lebih tua dan menolong teman yang kesulitan. Peribahasa yang tepat untuk sikap Rina?",
        solution_steps=[
            "Identifikasi sikap: menghormati, menolong, baik hati",
            "Pola: kebaikan yang ditabur → menuai kebaikan",
            "Atau: siapa menabur kebaikan akan menuai kebaikan",
        ],
        answer="siapa menabur kebaikan akan menuai kebaikan",
        explanation_why="Rina berbuat baik (menolong, menghormati). Peribahasa tentang kebaikan: siapa menabur kebaikan akan menuai kebaikan. Pola: kebaikan → balasan kebaikan.",
        question_type='peribahasa',
    ))

    lessons.add(TeachingLesson(
        problem="Budi dan Andi selalu tolong-menolong, baik saat senang maupun susah. Peribahasa untuk persahabatan mereka?",
        solution_steps=[
            "Identifikasi: tolong-menolong dalam suka dan duka",
            "Pola: kerja sama saling membantu → kebersamaan",
            "Cocokkan: berat sama dipikul ringan sama dijinjing",
        ],
        answer="berat sama dipikul ringan sama dijinjing",
        explanation_why="Tolong-menolong dalam suka dan duka adalah kerja sama sejati. Peribahasa: berat sama dipikul ringan sama dijinjing. Pola: cooperation → mutual support.",
        question_type='peribahasa',
    ))

    # ── BAHASA KIASAN: expanded from 3→7 — need subtype diversity ──
    lessons.add(TeachingLesson(
        problem="Bintang-bintang berkelipkan mata di langit malam yang gelap. Kata 'berkelipkan mata' termasuk majas....",
        solution_steps=[
            "Identifikasi subjek: bintang (benda mati)",
            "Identifikasi kata kerja: berkelipkan mata (sifat manusia)",
            "Benda mati diberi sifat manusia = personifikasi",
        ],
        answer="personifikasi",
        explanation_why="Personifikasi: benda mati (bintang) diberi sifat manusia (berkelipkan mata). Pola: non-human + human action → personifikasi.",
        question_type='bahasa_kiasan',
    ))

    lessons.add(TeachingLesson(
        problem="Pohon-pohon merunduk sedih ketika musim gugur tiba. Kata 'merunduk sedih' termasuk majas....",
        solution_steps=[
            "Identifikasi subjek: pohon (benda mati)",
            "Identifikasi kata kerja: merunduk sedih (sifat manusia)",
            "Benda mati diberi sifat manusia = personifikasi",
        ],
        answer="personifikasi",
        explanation_why="Personifikasi: benda mati (pohon) diberi sifat manusia (merunduk sedih). Pola: non-human + human emotion → personifikasi.",
        question_type='bahasa_kiasan',
    ))

    lessons.add(TeachingLesson(
        problem="Awan menangis di atas kota sore itu. Kata 'menangis' termasuk majas....",
        solution_steps=[
            "Identifikasi subjek: awan (benda mati)",
            "Identifikasi kata kerja: menangis (sifat manusia)",
            "Benda mati diberi sifat manusia = personifikasi",
        ],
        answer="personifikasi",
        explanation_why="Personifikasi: benda mati (awan) diberi sifat manusia (menangis). Pola: non-human + human action → personifikasi.",
        question_type='bahasa_kiasan',
    ))

    lessons.add(TeachingLesson(
        problem="Wajahnya bagaikan bulan purnama yang bersinar terang. Kata 'bagaikan bulan purnama' termasuk majas....",
        solution_steps=[
            "Identifikasi kata pembanding: bagaikan",
            "Bagaikan = perbandingan eksplisit",
            "Perbandingan eksplisit dengan kata pembanding = simile/perumpamaan",
        ],
        answer="perumpamaan (simile)",
        explanation_why="Simile: perbandingan eksplisit menggunakan kata pembanding 'bagaikan'. Pola: A + kata pembanding + B → simile. Kata pembanding: seperti, bagaikan, laksana, bak.",
        question_type='bahasa_kiasan',
    ))

    # ── IDE POKOK: expanded from 2→6 ──
    lessons.add(TeachingLesson(
        problem="Indonesia memiliki keragaman budaya yang sangat kaya. Setiap daerah mempunyai bahasa dan pakaian adat yang berbeda. Keragaman ini menjadikan Indonesia unik. Apa ide pokok paragraf tersebut?",
        solution_steps=[
            "Baca kalimat pertama: 'Indonesia memiliki keragaman budaya yang sangat kaya'",
            "Kalimat pertama menyatakan gagasan utama tentang keragaman budaya",
            "Kalimat berikutnya menjelaskan detail keragaman tersebut",
            "Kesimpulan: kalimat pertama = ide pokok",
        ],
        answer="keragaman budaya Indonesia",
        explanation_why="Dalam paragraf deduktif, ide pokok ada di kalimat pertama. Kalimat pertama menyatakan gagasan utama (keragaman budaya), kalimat lain menjelaskan. Pola: kalimat pertama = ide pokok.",
        question_type='ide_pokok',
    ))

    lessons.add(TeachingLesson(
        problem="Teknologi informasi berkembang sangat pesat. Internet memudahkan akses pengetahuan. Media sosial menghubungkan orang di seluruh dunia. Apa gagasan utama paragraf tersebut?",
        solution_steps=[
            "Baca kalimat pertama: 'Teknologi informasi berkembang sangat pesat'",
            "Kalimat pertama menyatakan gagasan utama",
            "Kalimat berikutnya memberikan contoh: internet dan media sosial",
            "Kesimpulan: kalimat pertama = ide pokok",
        ],
        answer="perkembangan teknologi informasi",
        explanation_why="Paragraf deduktif: ide pokok di awal. Kalimat pertama tentang perkembangan teknologi, sisanya penjelasan. Pola: kalimat pertama = gagasan utama.",
        question_type='ide_pokok',
    ))

    lessons.add(TeachingLesson(
        problem="Banyak siswa kurang minum air putih saat di sekolah. Mereka lebih memilih minuman manis. Oleh karena itu, kita harus membiasakan minum air putih yang cukup. Apa ide pokok paragraf tersebut?",
        solution_steps=[
            "Identifikasi kata penanda kesimpulan: 'Oleh karena itu'",
            "Kalimat dengan penanda kesimpulan berisi ide pokok",
            "Kesimpulan: ide pokok ada di kalimat terakhir",
            "Ide pokok: kita harus membiasakan minum air putih yang cukup",
        ],
        answer="kita harus membiasakan minum air putih yang cukup",
        explanation_why="Paragraf induktif: ide pokok di akhir, ditandai 'oleh karena itu'. Pola: penanda kesimpulan (oleh karena itu, jadi, maka) → ide pokok di akhir.",
        question_type='ide_pokok',
    ))

    lessons.add(TeachingLesson(
        problem="Pencemaran sungai semakin parah di kota besar. Limbah industri dan domestik mencemari air. Oleh karena itu, upaya pembersihan sungai harus segera dilakukan. Apa gagasan utama paragraf tersebut?",
        solution_steps=[
            "Identifikasi kata penanda kesimpulan: 'Oleh karena itu'",
            "Kalimat akhir berisi kesimpulan = gagasan utama",
            "Gagasan utama: upaya pembersihan sungai harus segera dilakukan",
        ],
        answer="upaya pembersihan sungai harus segera dilakukan",
        explanation_why="Paragraf induktif: ide pokok di akhir. Kata 'oleh karena itu' menandakan kesimpulan yang merupakan gagasan utama. Pola: karena itu/jadi/maka → ide pokok di akhir.",
        question_type='ide_pokok',
    ))

    # ── IMPLISIT: expanded from 1→5 ──
    lessons.add(TeachingLesson(
        problem="Gempa bumi menghancurkan jembatan penghubung dua kota. Truk pengangkut barang tidak bisa lewat. Harga kebutuhan pokok naik. Mengapa harga kebutuhan naik?",
        solution_steps=[
            "Identifikasi rantai sebab-akibat: gempa → jembatan hancur → truk tidak lewat → barang tidak sampai → harga naik",
            "Cari root cause: gempa bumi (pemicu awal)",
            "Jawaban: karena gempa menghancurkan jembatan sehingga distribusi terganggu",
        ],
        answer="gempa bumi menghancurkan jembatan sehingga distribusi barang terganggu",
        explanation_why="Implisit: A→B→C→D chain. Gempa → jembatan hancur → truk tidak lewat → harga naik. Root cause = gempa. Pola: cari peristiwa pertama dalam rantai.",
        question_type='implisit',
    ))

    lessons.add(TeachingLesson(
        problem="Virus menyebar cepat di kota itu. Banyak karyawan sakit sehingga perusahaan menghentikan produksi sementara. Mengapa produksi dihentikan?",
        solution_steps=[
            "Identifikasi rantai: virus → karyawan sakit → produksi berhenti",
            "Root cause: virus menyebar",
            "Jawaban: karena karyawan sakit akibat virus",
        ],
        answer="karyawan sakit akibat penyebaran virus",
        explanation_why="Implisit: virus → karyawan sakit → produksi berhenti. Pertanyaan menanyakan penyebab langsung (karyawan sakit) yang disebabkan oleh root cause (virus). Pola: A→B→C, pertanyaan tentang B atau C.",
        question_type='implisit',
    ))

    lessons.add(TeachingLesson(
        problem="Kebakaran hutan menghasilkan asap tebal. Asap menutupi bandara sehingga pesawat tidak bisa mendarat. Penumpang tertunda selama berjam-jam. Mengapa penumpang tertunda?",
        solution_steps=[
            "Identifikasi rantai: kebakaran → asap → bandara tertutup → pesawat tidak mendarat → penumpang tertunda",
            "Penyebab langsung: pesawat tidak bisa mendarat karena asap",
            "Root cause: kebakaran hutan",
        ],
        answer="pesawat tidak bisa mendarat karena asap dari kebakaran hutan",
        explanation_why="Implisit: kebakaran → asap → bandara tertutup → pesawat tidak mendarat → tertunda. Pola: A→B→C→D→E, cari link yang menghubungkan ke pertanyaan.",
        question_type='implisit',
    ))

    lessons.add(TeachingLesson(
        problem="Kekeringan melanda desa itu. Tanaman padi layu dan petani tidak bisa panen. Harga beras di pasar naik drastis. Mengapa harga beras naik?",
        solution_steps=[
            "Identifikasi rantai: kekeringan → tanaman layu → tidak bisa panen → beras langka → harga naik",
            "Root cause: kekeringan",
            "Penyebab langsung: beras langka karena panen gagal",
        ],
        answer="panen gagal karena kekeringan sehingga beras langka dan harga naik",
        explanation_why="Implisit: kekeringan → tanaman layu → panen gagal → beras langka → harga naik. Pola: A→B→C→D→E chain, cari link ke pertanyaan.",
        question_type='implisit',
    ))

    # ── PERBANDINGAN: expanded from 4→8 ──
    lessons.add(TeachingLesson(
        problem="Sari berolahraga setiap pagi dan makan makanan sehat. Dina jarang berolahraga dan suka makan junk food. Apa perbedaan gaya hidup Sari dan Dina?",
        solution_steps=[
            "Identifikasi tindakan Sari: olahraga rutin, makan sehat",
            "Identifikasi tindakan Dina: jarang olahraga, makan junk food",
            "Abstraksi: sehat vs tidak sehat",
        ],
        answer="sehat vs tidak sehat",
        explanation_why="Perbandingan: ekstrak KUALITAS ABSTRAK, bukan tindakan literal. Sari = sehat (olahraga + makan sehat). Dina = tidak sehat (jarang olahraga + junk food). Pola: compare actions → abstract quality.",
        question_type='perbandingan',
    ))

    lessons.add(TeachingLesson(
        problem="Eko selalu mengerjakan PR tepat waktu dan membuat jadwal belajar. Fajar sering menunda PR dan bermain game hingga larut. Apa perbedaan kebiasaan belajar Eko dan Fajar?",
        solution_steps=[
            "Identifikasi kebiasaan Eko: tepat waktu, terjadwal",
            "Identifikasi kebiasaan Fajar: menunda, bermain game",
            "Abstraksi: rajin/disiplin vs malas/prokrastinasi",
        ],
        answer="rajin/disiplin vs malas/prokrastinasi",
        explanation_why="Perbandingan: Eko = rajin (tepat waktu, terjadwal). Fajar = malas (menunda, game). Pola: compare methods → abstract quality difference.",
        question_type='perbandingan',
    ))

    lessons.add(TeachingLesson(
        problem="Andi lebih tinggi dari Budi. Siapa yang lebih pendek?",
        solution_steps=[
            "Identifikasi relasi: Andi > Budi (tinggi)",
            "Pertanyaan bertanya yang LEBIH PENDEK",
            "Jika Andi lebih tinggi, maka Budi lebih pendek",
            "Jawaban: pihak yang BERLAWANAN dari yang disebut",
        ],
        answer="Budi",
        explanation_why="Perbandingan terbalik: 'lebih tinggi' → lawannya 'lebih pendek'. Jika A lebih X dari B, maka B lebih (lawan X) dari A.",
        question_type='perbandingan',
    ))

    lessons.add(TeachingLesson(
        problem="Raja sangat kaya dan berkuasa tetapi dia tidur di lantai dan makan seadanya. Kehidupan raja sebenarnya?",
        solution_steps=[
            "Temukan kata kontras: 'tetapi'",
            "Sebelum tetapi: kaya dan berkuasa (kesan awal)",
            "Setelah tetapi: tidur di lantai, makan seadanya (kenyataan)",
            "Kehidupan sebenarnya = sederhana",
        ],
        answer="sederhana",
        explanation_why="Kontras 'tetapi': setelah tetapi adalah keadaan sebenarnya. Tidur di lantai + makan seadanya = hidup sederhana. Kontras mengoreksi kesan awal.",
        question_type='perbandingan',
    ))

    # ── EKSPLISIT: expanded from 2→5 ──
    lessons.add(TeachingLesson(
        problem="Taman Nasional Komodo terletak di Provinsi Nusa Tenggara Timur. Taman ini terkenal dengan hewan endemiknya yaitu komodo. Di provinsi mana Taman Nasional Komodo terletak?",
        solution_steps=[
            "Identifikasi pertanyaan: 'di provinsi mana'",
            "Cari provinsi di teks: Nusa Tenggara Timur",
            "Jawaban langsung dari teks",
        ],
        answer="Nusa Tenggara Timur",
        explanation_why="Pertanyaan 'di provinsi mana' → cari nama provinsi yang disebut di teks. Nusa Tenggara Timur adalah provinsi yang disebut. Pola: where → find location in text.",
        question_type='eksplisit',
    ))

    lessons.add(TeachingLesson(
        problem="Sekolah Dasar Negeri 3 Surabaya didirikan pada tahun 1975 oleh Bapak Suwiryo. Sekolah ini memiliki 12 ruang kelas. Siapa yang mendirikan SDN 3 Surabaya?",
        solution_steps=[
            "Identifikasi pertanyaan: 'siapa'",
            "Cari nama orang di teks: Bapak Suwiryo",
            "Jawaban langsung dari teks",
        ],
        answer="Bapak Suwiryo",
        explanation_why="Pertanyaan 'siapa' → cari nama orang di teks. Bapak Suwiryo mendirikan sekolah. Pola: who → find person name in text.",
        question_type='eksplisit',
    ))

    lessons.add(TeachingLesson(
        problem="Perpustakaan kota buka setiap hari Senin sampai Sabtu pukul 09.00 sampai 17.00. Pada hari Minggu perpustakaan tutup. Pada pukul berapa perpustakaan kota buka?",
        solution_steps=[
            "Identifikasi pertanyaan: 'pukul berapa'",
            "Cari waktu di teks: 09.00 sampai 17.00",
            "Pukul buka = 09.00",
        ],
        answer="09.00",
        explanation_why="Pertanyaan 'pukul berapa' → cari waktu di teks. 09.00 adalah jam buka. Pola: when/what time → find time value in text.",
        question_type='eksplisit',
    ))

    # ── SIKAP TOKOH: expanded from 2→4 ──
    lessons.add(TeachingLesson(
        problem="Rani selalu menghormati orang yang lebih tua dan menolong teman yang kesulitan. Ia dikenal sebagai anak yang baik hati. Apa sikap Rani?",
        solution_steps=[
            "Identifikasi tindakan: menghormati, menolong",
            "Identifikasi sifat: baik hati",
            "Menghormati + menolong = baik hati / peduli",
        ],
        answer="baik hati atau peduli",
        explanation_why="Sikap tokoh: menghormati + menolong → baik hati/peduli. Pola: tindakan positif terhadap orang lain = sikap sosial yang baik.",
        question_type='sikap_tokoh',
    ))

    lessons.add(TeachingLesson(
        problem="Meskipun dicemooh, Amir tetap berbuat baik dan menolong siapa saja. Sifat Amir?",
        solution_steps=[
            "Identifikasi tindakan: berbuat baik meskipun dicemooh",
            "Tetap baik meskipun diperlakukan buruk = pemaaf/ikhlas",
            "Sifat: pemaaf, ikhlas, atau tabah",
        ],
        answer="pemaaf atau ikhlas",
        explanation_why="Berbuat baik meskipun dicemooh = pemaaf/ikhlas/tabah. Pola: tindakan baik terus meskipun diperlakukan buruk = pemaaf.",
        question_type='sikap_tokoh',
    ))

    # ── TEKS EKSPLANASI: expanded from 2→4 ──
    lessons.add(TeachingLesson(
        problem="Gunung meletus mengeluarkan lava dan abu vulkanik. Lava membanjiri desa di lereng. Abu vulkanik menutupi lahan pertanian. Mengapa pertanian rusak?",
        solution_steps=[
            "Identifikasi rantai: letusan → lava + abu → desa terbanjiri + pertanian tertutup",
            "Penyebab langsung: abu vulkanik menutupi lahan",
            "Root cause: gunung meletus",
        ],
        answer="abu vulkanik menutupi lahan pertanian akibat letusan gunung",
        explanation_why="Teks eksplanasi: letusan → abu → pertanian rusak. Pola: proses alam A→B→C, cari link langsung ke pertanyaan.",
        question_type='teks_eksplanasi',
    ))

    lessons.add(TeachingLesson(
        problem="Erosi terjadi karena hujan mengikis tanah yang tidak tertutup vegetasi. Akar pohon tidak lagi menahan tanah setelah penebangan. Mengapa erosi terjadi?",
        solution_steps=[
            "Identifikasi sebab: hujan + tanah tanpa vegetasi + akar tidak menahan",
            "Proses: penebangan → tidak ada akar → tanah longgar → hujan mengikis → erosi",
            "Root cause: penebangan hutan (menghilangkan vegetasi)",
        ],
        answer="hujan mengikis tanah yang tidak lagi ditahan oleh akar pohon",
        explanation_why="Erosi: penebangan → tidak ada akar → tanah longgar → hujan mengikis. Pola: proses sebab-akibat berantai dalam alam.",
        question_type='teks_eksplanasi',
    ))

    return lessons
