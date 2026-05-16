"""
Training Corpora — Rich Indonesian language patterns for AAM training.

This corpus is designed to exercise the full feedback loop:
- Transaction patterns (menjual ke, memberikan kepada, mengirim ke)
- Causal patterns (karena, sehingga, akibat)
- Purpose patterns (untuk, agar, supaya)
- Negation patterns (tidak, bukan, jangan)
- Passive/active voice patterns
- Ambiguous pronoun patterns (dia, mereka, ini, itu)
- Various semantic frames from different domains

Each pattern is presented in multiple variations so the system can:
1. Detect gaps in the first occurrence
2. Learn from corrections
3. Recognize patterns in subsequent occurrences
4. Generalize to new instances
"""

from typing import Dict, List


class TrainingCorpus:
    """Collection of training corpora for AAM ingest training."""

    # ────────────────────────────────────────────────────────────────
    # Transaction Patterns — Menjual/Membeli (Buy/Sell)
    # ────────────────────────────────────────────────────────────────

    TRANSACTION_SELL = [
        "Budi menjual barang ke saya",
        "PT ABC menjual mesin ke vendor kami",
        "Ibu menjual kue ke tetangga",
        "Pak Ahmad menjual rumah kepada pembeli dari Jakarta",
        "Perusahaan menjual saham ke investor asing",
        "Toko itu menjual buah ke pelanggan setia",
        "Dia menjual motor ke teman kuliahnya",
        "Mereka menjual hasil panen ke pasar",
        "Warung itu menjual nasi goreng ke pelanggan",
        "Bank menjual obligasi ke nasabah korporat",
        "Agen menjual asuransi ke klien baru",
        "Pedagang menjual ikan ke restoran",
        "Galerry menjual lukisan ke kolektor seni",
        "Developer menjual apartemen ke pemilik pertama",
        "Pabrik menjual produk ke distributor",
    ]

    TRANSACTION_BUY = [
        "Saya membeli buku dari toko itu",
        "Kami membeli bahan baku dari supplier",
        "Dia membeli mobil dari dealer resmi",
        "Mereka membeli tanah dari warga desa",
        "Perusahaan membeli peralatan dari vendor luar negeri",
        "Ibu membeli sayur dari pasar tradisional",
        "Pak Budi membeli laptop dari toko online",
        "Kami membeli rumah dari developer properti",
    ]

    TRANSACTION_GIVE = [
        "Ibu memberikan uang kepada anaknya",
        "Pemerintah memberikan bantuan kepada warga terdampak",
        "Dia memberikan hadiah kepada pacarnya",
        "Perusahaan memberikan bonus kepada karyawan",
        "Guru memberikan tugas kepada murid",
        "Dokter memberikan resep kepada pasien",
        "Mereka memberikan donasi kepada yayasan",
        "Atasan memberikan arahan kepada tim",
    ]

    TRANSACTION_SEND = [
        "Kantor mengirim surat ke kantor cabang",
        "Dia mengirim paket kepada keluarganya",
        "Perusahaan mengirim barang ke gudang",
        "Bank mengirim notifikasi ke nasabah",
        "Kami mengirim laporan kepada manajemen",
        "Toko mengirim pesanan ke alamat pembeli",
        "Pemerintah mengirim tim medis ke daerah terpencil",
        "Mereka mengirim undangan kepada semua tamu",
    ]

    # ────────────────────────────────────────────────────────────────
    # Causal Patterns — Karena/Sebab/Akibat
    # ────────────────────────────────────────────────────────────────

    CAUSAL_KARENA = [
        "Raymond membuat aplikasi karena proses manual lambat",
        "Dia belajar keras karena ujian besok",
        "Mereka pindah rumah karena banjir",
        "Perusahaan bangkrut karena salah kelola",
        "Ibu marah karena anak tidak mau makan",
        "Toko tutup karena rugi terus menerus",
        "Dia sakit karena terlalu banyak bekerja",
        "Pesawat terlambat karena cuaca buruk",
        "Proyek gagal karena kurang dana",
        "Siswa drop out karena masalah finansial",
    ]

    CAUSAL_SEHINGGA = [
        "Hujan deras sehingga jalanan banjir",
        "Dia belajar tekun sehingga nilai bagus",
        "Harga naik sehingga daya beli turun",
        "Teknologi maju sehingga kehidupan berubah",
        "Internet lambat sehingga meeting terganggu",
        "Kurang tidur sehingga konsentrasi menurun",
        "Kebijakan baru sehingga pasar bergejolak",
        "Data hilang sehingga laporan tertunda",
    ]

    # ────────────────────────────────────────────────────────────────
    # Purpose Patterns — Untuk/Agar/Supaya
    # ────────────────────────────────────────────────────────────────

    PURPOSE_UNTUK = [
        "Dia berolahraga untuk menjaga kesehatan",
        "Perusahaan mengadakan pelatihan untuk meningkatkan skill karyawan",
        "Ibu menabung untuk biaya pendidikan anak",
        "Pemerintah membangun jembatan untuk menghubungkan desa terpencil",
        "Mereka berkumpul untuk merayakan ulang tahun",
        "Tim melakukan riset untuk menemukan obat baru",
        "Saya belajar programming untuk mendapatkan pekerjaan bagus",
        "Kami menghemat listrik untuk mengurangi biaya",
    ]

    PURPOSE_AGAR = [
        "Dia belajar agar lulus ujian",
        "Ibu memasak agar anak mau makan",
        "Perusahaan berinovasi agar tetap kompetitif",
        "Kita harus hemat agar cukup sampai akhir bulan",
        "Mereka berkomunikasi rutin agar proyek lancar",
        "Guru menjelaskan ulang agar siswa paham",
    ]

    # ────────────────────────────────────────────────────────────────
    # Location Patterns — Di/Ke/Dari
    # ────────────────────────────────────────────────────────────────

    LOCATION_DI = [
        "Dia bekerja di kantor pusat",
        "Mereka tinggal di Bandung",
        "Meeting diadakan di ruang konferensi",
        "Barang disimpan di gudang belakang",
        "Anak bermain di taman kota",
        "Pasar tradisional berada di pusat kota",
    ]

    # ────────────────────────────────────────────────────────────────
    # Instrument Patterns — Dengan/Memakai
    # ────────────────────────────────────────────────────────────────

    INSTRUMENT_DENGAN = [
        "Dia menulis surat dengan tangan",
        "Ibu memasak nasi dengan rice cooker",
        "Mereka berkomunikasi dengan email",
        "Pekerja memotong kayu dengan gergaji",
        "Dokter mengobati pasien dengan antibiotik",
        "Programmer membuat aplikasi dengan Python",
        "Seniman melukis dengan cat minyak",
        "Siswa mengerjakan tugas dengan komputer",
    ]

    # ────────────────────────────────────────────────────────────────
    # Negation Patterns
    # ────────────────────────────────────────────────────────────────

    NEGATION_TIDAK = [
        "Dia tidak mau makan karena sedang marah",
        "Perusahaan tidak membayar utang karena bangkrut",
        "Budi tidak datang karena sakit",
        "Mereka tidak setuju karena alasan tertentu",
        "Proyek tidak selesai karena kurang sumber daya",
    ]

    NEGATION_BUKAN = [
        "Itu bukan kesalahan dia",
        "Bukan uang yang penting tapi kebahagiaan",
        "Dia bukan mahasiswa tapi dosen",
        "Ini bukan masalah teknis tapi masalah manusia",
    ]

    # ────────────────────────────────────────────────────────────────
    # Ambiguous Pronoun Patterns
    # ────────────────────────────────────────────────────────────────

    AMBIGUOUS_PRONOUNS = [
        "Budi memberi buku kepada dia",
        "Mereka pergi ke kantor bersama",
        "Ini sangat penting untuk kita semua",
        "Itu bukan masalah dia",
        "Dia berkata bahwa dia akan datang",
        "Mereka bilang mereka setuju",
        "Kami percaya mereka bisa menyelesaikannya",
        "Saya bertemu dia di toko",
    ]

    # ────────────────────────────────────────────────────────────────
    # Complex Multi-Clause Patterns
    # ────────────────────────────────────────────────────────────────

    COMPLEX_SENTENCES = [
        "Perusahaan mengimplementasikan sistem baru karena yang lama sudah tidak efisien",
        "Dia membeli laptop baru untuk mengerjakan tugas akhir karena yang lama rusak",
        "Tim membangun pipeline data agar analisis bisa berjalan otomatis",
        "Pemerintah memberikan subsidi kepada petani karena harga pupuk naik",
        "Bank mengirim peringatan kepada nasabah agar segera membayar cicilan",
        "Guru memberikan pekerjaan rumah kepada siswa untuk mengukur pemahaman",
        "Perusahaan mengirim delegasi ke konferensi untuk mempresentasikan riset",
        "Mereka membangun shelter untuk pengungsi karena bencana melanda",
        "Dokter meresepkan obat kepada pasien agar cepat sembuh",
        "Manajer memberikan bonus kepada tim karena target tercapai",
    ]

    # ────────────────────────────────────────────────────────────────
    # Passive Voice Patterns
    # ────────────────────────────────────────────────────────────────

    PASSIVE_VOICE = [
        "Barang dijual oleh Budi kepada saya",
        "Laporan dibuat oleh tim audit untuk manajemen",
        "Surat dikirim oleh kantor ke cabang",
        "Proyek dikerjakan oleh tim IT karena mendesak",
        "Hadiah diberikan oleh ibu kepada anak",
        "Keputusan diambil oleh direktur karena mendesak",
        "Obat diresepkan oleh dokter kepada pasien",
        "Tugas diberikan oleh guru kepada siswa",
    ]

    # ────────────────────────────────────────────────────────────────
    # Business Domain
    # ────────────────────────────────────────────────────────────────

    BUSINESS = [
        "Direktur menyetujui proposal karena sudah direvisi",
        "Finance mengirim invoice ke klien untuk pembayaran",
        "HR memberikan pelatihan kepada karyawan baru agar produktif",
        "Marketing meluncurkan kampanye untuk meningkatkan brand awareness",
        "Sales menjual produk ke pelanggan potensial karena permintaan tinggi",
        "IT membangun infrastruktur cloud agar sistem scalable",
        "CEO mengumumkan restrukturisasi karena kinerja perusahaan menurun",
        "Vendor mengirim barang ke gudang untuk persediaan",
        "Auditor memeriksa laporan keuangan karena ada indikasi penyimpangan",
        "CFO mengajukan budget kepada board untuk ekspansi bisnis",
    ]

    # ────────────────────────────────────────────────────────────────
    # Academic/Education Domain
    # ────────────────────────────────────────────────────────────────

    ACADEMIC = [
        "Profesor memberikan kuliah kepada mahasiswa karena jadwal sudah ditentukan",
        "Mahasiswa mengerjakan skripsi untuk memenuhi syarat kelulusan",
        "Peneliti mempublikasikan paper agar hasil riset diketahui",
        "Universitas memberikan beasiswa kepada mahasiswa berprestasi",
        "Laboratorium membeli peralatan baru karena yang lama usang",
        "Dosen membimbing mahasiswa agar penelitian berkualitas",
        "Perpustakaan meminjamkan buku kepada anggota untuk dibaca",
        "Sekolah mengirim siswa ke kompetisi agar berpengalaman",
    ]

    # ────────────────────────────────────────────────────────────────
    # Technology Domain
    # ────────────────────────────────────────────────────────────────

    TECHNOLOGY = [
        "Engineer membangun microservice karena monolith sudah tidak scalable",
        "DevOps meng deploy aplikasi ke server production untuk release",
        "Data Scientist membangun model ML untuk memprediksi churn",
        "Tim mengimplementasikan CI/CD agar deployment cepat",
        "Backend developer membuat API untuk frontend consumption",
        "QA melakukan testing karena bug critical ditemukan",
        "CTO memutuskan migrasi ke cloud karena on-premise mahal",
        "Team lead memberikan code review kepada developer agar kualitas terjaga",
    ]

    # ────────────────────────────────────────────────────────────────
    # Get All Training Sentences
    # ────────────────────────────────────────────────────────────────

    def get_all(self) -> List[str]:
        """Get all training sentences."""
        all_sentences = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, list) and all(isinstance(s, str) for s in attr):
                all_sentences.extend(attr)
        return all_sentences

    def get_by_category(self, category: str) -> List[str]:
        """Get training sentences by category name."""
        attr = getattr(self, category, None)
        if attr and isinstance(attr, list):
            return attr
        return []

    def get_categories(self) -> Dict[str, int]:
        """Get all categories and their sentence counts."""
        categories = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, list) and all(isinstance(s, str) for s in attr):
                categories[attr_name] = len(attr)
        return categories

    def get_transaction_all(self) -> List[str]:
        """Get all transaction-related sentences."""
        return (self.TRANSACTION_SELL + self.TRANSACTION_BUY +
                self.TRANSACTION_GIVE + self.TRANSACTION_SEND)

    def get_causal_all(self) -> List[str]:
        """Get all causal-related sentences."""
        return self.CAUSAL_KARENA + self.CAUSAL_SEHINGGA

    def get_purpose_all(self) -> List[str]:
        """Get all purpose-related sentences."""
        return self.PURPOSE_UNTUK + self.PURPOSE_AGAR

    def get_progressive_curriculum(self) -> List[str]:
        """
        Get a progressive curriculum — ordered from simple to complex.

        The system ingests simple sentences first, learns basic patterns,
        then moves to more complex ones. This mirrors how a child learns:
        simple transactions → causal reasoning → purpose → ambiguity.
        """
        curriculum = []
        # Phase 1: Simple transactions (learn basic SVO)
        curriculum.extend(self.TRANSACTION_SELL[:5])
        curriculum.extend(self.TRANSACTION_BUY[:3])
        # Phase 2: Give/Send (learn Recipient role)
        curriculum.extend(self.TRANSACTION_GIVE[:5])
        curriculum.extend(self.TRANSACTION_SEND[:5])
        # Phase 3: Causal reasoning (learn Cause role)
        curriculum.extend(self.CAUSAL_KARENA[:5])
        curriculum.extend(self.CAUSAL_SEHINGGA[:3])
        # Phase 4: Purpose (learn Purpose role)
        curriculum.extend(self.PURPOSE_UNTUK[:5])
        curriculum.extend(self.PURPOSE_AGAR[:3])
        # Phase 5: Location & Instrument
        curriculum.extend(self.LOCATION_DI[:3])
        curriculum.extend(self.INSTRUMENT_DENGAN[:3])
        # Phase 6: Negation
        curriculum.extend(self.NEGATION_TIDAK[:3])
        curriculum.extend(self.NEGATION_BUKAN[:2])
        # Phase 7: Ambiguous pronouns
        curriculum.extend(self.AMBIGUOUS_PRONOUNS[:3])
        # Phase 8: Complex multi-clause
        curriculum.extend(self.COMPLEX_SENTENCES[:5])
        # Phase 9: Passive voice
        curriculum.extend(self.PASSIVE_VOICE[:3])
        # Phase 10: Domain-specific
        curriculum.extend(self.BUSINESS[:5])
        curriculum.extend(self.ACADEMIC[:3])
        curriculum.extend(self.TECHNOLOGY[:3])
        return curriculum
