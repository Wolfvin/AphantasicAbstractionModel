# @WHO:   self-ai/src/unconscious/training_pairs_dataset.py
# @WHAT:  Diverse training pairs for ProjectionTrainer — 80+ pairs across 10 domains
# @PART:  self-ai/unconscious
# @ENTRY: get_training_pairs()

"""Training pairs dataset for ProjectionTrainer.

Each pair: (experience_text, correct_output_text)
  - experience_text: a reasoning principle or insight (bge-m3 will embed this)
  - correct_output_text: what the model should output when this insight is active

Diversity requirement: 10+ domains, 8 pairs each, minimum 80 total.
All texts in Bahasa Indonesia unless the domain inherently uses English terms.

Why diverse pairs matter:
  Training with only 10 domain-specific pairs causes severe overfitting —
  the projection learns to map those exact embeddings but fails on anything
  else. With 80+ pairs from diverse domains, the projection learns a
  GENERAL mapping from bge-m3 space to Qwen3 hidden state space that
  transfers across domains.
"""


def get_training_pairs() -> list[tuple[str, str]]:
    """Return diverse training pairs for ProjectionTrainer.

    Each pair: (experience_text, correct_output_text)
    - experience_text: a reasoning principle or insight (bge-m3 will embed this)
    - correct_output_text: what the model should output when this insight is active

    Diversity requirement: 10+ domains, 8 pairs each, minimum 80 total.
    """
    pairs = [

        # ═══════════════════════════════════════════════════════════════
        # Domain 1: Pengecualian / exclusion logic
        # Kata kecuali, selain, tidak termasuk menandai elemen
        # yang dikecualikan dari kelompok umum.
        # ═══════════════════════════════════════════════════════════════

        ("kata kecuali menandai elemen yang tidak termasuk dalam kelompok umum",
         "harimau tidak termasuk karena kata kecuali mengeluarkannya dari kelompok"),

        ("kata selain menunjukkan bahwa sesuatu berada di luar kategori yang disebut",
         "selain ayam berarti ayam tidak termasuk dalam kelompok yang dimaksud"),

        ("dalam logika pengecualian jawaban selalu merujuk pada entitas yang dikecualikan",
         "budi adalah jawabannya karena ia dikecualikan dari kelompok yang lulus"),

        ("kata terkecuali memiliki fungsi yang sama dengan kecuali yaitu menandai pengecualian",
         "yang terkecuali adalah entitas yang tidak termasuk dalam aturan umum"),

        ("setelah kata kecuali entitas yang menyusul adalah pengecualian dari aturan sebelumnya",
         "setelah kecuali disebutkan maka entitas tersebut berada di luar kelompok utama"),

        ("kata tidak termasuk mengindikasikan bahwa suatu entitas dikeluarkan dari himpunan",
         "entitas yang tidak termasuk adalah jawaban karena ia berada di luar himpunan"),

        ("kata kecuali membalikkan keanggotaan kelompok semua menjadi tidak dan tidak menjadi ya",
         "kecuali membalik status keanggotaan sehingga yang dikecualikan berstatus berlawanan"),

        ("dalam pola pengecualian ganda setiap kecuali menambah satu pengecualian baru pada daftar",
         "dua kata kecuali berarti ada dua entitas yang dikeluarkan dari kelompok utama"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 2: Operasi matematika
        # Pengurangan, pembagian, persentase, perkalian —
        # pola perhitungan dasar.
        # ═══════════════════════════════════════════════════════════════

        ("operasi pengurangan mengurangi nilai awal dengan nilai yang dikurangkan",
         "50 dikurangi 15 sama dengan 35 karena pengurangan mengurangi jumlah awal"),

        ("kata dikurangi berarti menghitung selisih antara dua bilangan",
         "selisih antara 80 dan 30 adalah 50 karena dikurangi berarti mengurangi"),

        ("pembagian membagi suatu jumlah menjadi bagian yang sama besar",
         "100 dibagi 4 sama dengan 25 karena pembagian membagi rata jumlah"),

        ("persentase menyatakan bagian per seratus dari suatu keseluruhan",
         "25 persen dari 200 adalah 50 karena persentase menghitung bagian per seratus"),

        ("perkalian mengulang penjumlahan sejumlah kali yang telah ditentukan",
         "6 kali 7 sama dengan 42 karena perkalian menjumlahkan 6 sebanyak 7 kali"),

        ("kata kembalian berarti menghitung selisih antara uang yang dibayar dan harga",
         "kembalian dari 100 untuk harga 75 adalah 25 karena selisihnya 25"),

        ("operasi hitung berurutan dikerjakan dari kiri ke kanan kecuali perkalian dan pembagian didahulukan",
         "2 ditambah 3 kali 4 sama dengan 14 karena perkalian dikerjakan lebih dulu"),

        ("kata setengah berarti membagi dua atau mengalikan dengan nol koma lima",
         "setengah dari 80 adalah 40 karena membagi dua menghasilkan separuh"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 3: Sebab-akibat
        # Karena, sehingga, akibatnya — hubungan kausal
        # antara peristiwa.
        # ═══════════════════════════════════════════════════════════════

        ("kata karena menunjukkan alasan atau penyebab dari suatu peristiwa yang terjadi",
         "hujan turun karena awan mendung menandakan hubungan sebab dan akibat"),

        ("kata sehingga menandakan konsekuensi atau hasil dari suatu kondisi",
         "dia belajar keras sehingga mendapat nilai bagus menunjukkan hasil usaha"),

        ("kata akibatnya menyatakan dampak yang timbul dari kejadian sebelumnya",
         "akibatnya banjir terjadi karena hujan lebat tanpa henti"),

        ("hubungan sebab-akibat menghubungkan peristiwa penyebab dengan peristiwa hasil",
         "karena tanah longsor maka jalan tertutup menunjukkan kausalitas"),

        ("kata menyebabkan menandakan bahwa subjek adalah agen penyebab dari suatu kejadian",
         "polusi menyebabkan penyakit pernapasan karena ada hubungan kausal di antaranya"),

        ("kata maka menunjukkan kesimpulan atau konsekuensi logis dari premis sebelumnya",
         "jika hujan maka tanah basah menunjukkan konsekuensi logis"),

        ("karena dan sebab keduanya menandai bagian penyebab dalam kalimat kausal",
         "sebab tidak makan maka lapar menunjukkan alasan dan akibatnya"),

        ("dalam rantai kausal satu akibat bisa menjadi sebab untuk akibat berikutnya",
         "hujan menyebabkan banjir dan banjir menyebabkan kerugian adalah rantai kausal"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 4: Perbandingan / analogi
        # Lebih besar dari, mirip dengan, berbeda karena —
        # hubungan komparatif.
        # ═══════════════════════════════════════════════════════════════

        ("kata lebih menandakan bahwa satu entitas melebihi entitas lain dalam sifat tertentu",
         "gajah lebih besar dari kucing berarti ukuran gajah melebihi kucing"),

        ("kata kurang menunjukkan bahwa satu entitas di bawah entitas lain dalam perbandingan",
         "5 kurang dari 8 berarti nilai 5 berada di bawah 8"),

        ("analogi membandingkan dua hal berbeda berdasarkan kesamaan pola atau hubungan",
         "hati seperti pompa adalah analogi karena pola fungsinya mirip"),

        ("kata mirip menunjukkan kesamaan parsial antara dua entitas yang tidak identik",
         "kucing mirip harimau karena keduanya karnivora tapi ukurannya berbeda"),

        ("kata berbeda menandakan adanya perbedaan mencolok antara dua hal yang dibandingkan",
         "katak berbeda dari ikan karena katak bernapas dengan paru-paru"),

        ("perbandingan selalu membutuhkan dua hal yang dikontraskan pada satu dimensi",
         "emas lebih berat dari aluminium menunjukkan perbandingan pada dimensi berat"),

        ("kata paling menandai tingkat tertinggi dalam perbandingan kelompok",
         "gunung everest paling tinggi berarti tidak ada yang melebihinya"),

        ("kata dibandingkan dengan memulai perbandingan terhadap standar acuan",
         "dibandingkan dengan besi aluminium lebih ringan menunjukkan perbedaan bobot"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 5: Urutan / sequence
        # Pertama, kemudian, akhirnya — pola temporal
        # dan prosedural.
        # ═══════════════════════════════════════════════════════════════

        ("kata pertama menandakan langkah awal atau awal urutan dalam proses",
         "pertama campur tepung dan air menunjukkan langkah awal resep"),

        ("kata kemudian menunjukkan langkah berikutnya setelah langkah sebelumnya selesai",
         "kemudian aduk adonan berarti langkah ini mengikuti langkah sebelumnya"),

        ("kata akhirnya menandai langkah terakhir atau kesimpulan dari urutan",
         "akhirnya panggang di oven menunjukkan langkah penutup proses"),

        ("kata selanjutnya menghubungkan langkah yang sedang berlangsung ke langkah berikutnya",
         "selanjutnya tambahkan gula berarti setelah langkah ini lakukan yang berikut"),

        ("urutan kronologis menyusun peristiwa berdasarkan waktu terjadinya dari awal ke akhir",
         "pagi sarapan lalu siang makan siang adalah urutan kronologis"),

        ("kata setelah itu menunjukkan bahwa suatu kejadian mengikuti kejadian sebelumnya",
         "setelah itu pergi ke sekolah berarti kegiatan ini mengikuti kegiatan sebelumnya"),

        ("kata sebelum menandakan kejadian yang terjadi lebih dahulu dari kejadian lain",
         "sebelum tidur sikat gigi berarti sikat gigi dilakukan lebih dulu"),

        ("dalam prosedur langkah tidak boleh dibalik karena setiap langkah bergantung pada hasil sebelumnya",
         "masak nasi sebelum menggoreng karena prosedur memasak berurutan"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 6: Identifikasi entitas
        # Siapa, apa, yang mana — pola pencarian
        # dan ekstraksi entitas dari teks.
        # ═══════════════════════════════════════════════════════════════

        ("kata siapa menanyakan identitas orang yang terlibat dalam suatu kejadian",
         "siapa presiden pertama merujuk pada tokoh yang memegang jabatan"),

        ("kata apa menanyakan benda kejadian atau konsep yang menjadi fokus pertanyaan",
         "apa ibu kota indonesia merujuk pada nama kota yang menjadi pusat pemerintahan"),

        ("kata dimana menanyakan lokasi tempat terjadinya suatu kejadian atau keberadaan entitas",
         "dimana monas berdiri merujuk pada lokasi geografis monumen tersebut"),

        ("kata yang mana meminta pemilihan satu entitas dari beberapa pilihan yang tersedia",
         "yang mana bilangan genap dari daftar berikut meminta identifikasi entitas"),

        ("identifikasi entitas memerlukan pencocokan deskripsi dalam pertanyaan dengan informasi di teks",
         "temukan nama guru yang mengajar matematika berarti cari entitas yang cocok"),

        ("kata berapa menanyakan jumlah atau nilai numerik dari suatu entitas",
         "berapa jumlah siswa di kelas menanyakan angka numerik pasti"),

        ("kata kapan menanyakan waktu terjadinya suatu peristiwa atau kejadian tertentu",
         "kapan indonesia merdeka merujuk pada tanggal kemerdekaan negara"),

        ("kata mengapa menanyakan alasan atau penyebab di balik suatu kejadian atau keadaan",
         "mengapa daun berwarna hijau menanyakan alasan di balik warna daun"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 7: Negasi
        # Bukan, tidak, jangan — pola penyangkalan
        # dan pembalikan makna.
        # ═══════════════════════════════════════════════════════════════

        ("kata tidak menyangkal pernyataan yang mengikutinya sehingga maknanya menjadi kebalikan",
         "tidak suka berarti lawan kata dari suka yaitu tidak menyukai"),

        ("kata bukan menolak identitas atau kategori yang disebutkan setelahnya",
         "bukan ikan berarti entitas tersebut tidak termasuk kategori ikan"),

        ("kata jangan merupakan perintah negatif yang melarang tindakan yang disebutkan",
         "jangan berlari berarti dilarang untuk berlari di situasi tersebut"),

        ("kata belum menyangkal bahwa sesuatu sudah terjadi tapi menyisakan kemungkinan di masa depan",
         "belum makan berarti saat ini tidak sudah makan tapi bisa nanti"),

        ("negasi ganda membatalkan satu sama lain sehingga makna kembali ke positif",
         "tidak tidak boleh berarti boleh karena dua negasi saling membatalkan"),

        ("kata tanpa menunjukkan ketiadaan atau kekurangan sesuatu yang biasanya ada",
         "kopi tanpa gula berarti kopi tersebut tidak mengandung gula"),

        ("kata tiada merupakan bentuk sastra dari tidak ada yang menandakan ketiadaan mutlak",
         "tiada yang bisa dilakukan berarti tidak ada cara yang tersedia"),

        ("kata bukan tapi membatalkan bagian sebelum tapi dan menegaskan bagian setelahnya",
         "bukan merah tapi biru berarti warna yang benar adalah biru bukan merah"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 8: Definisi / klasifikasi
        # Adalah, termasuk, merupakan — pola penetapan
        # identitas dan kategori.
        # ═══════════════════════════════════════════════════════════════

        ("kata adalah mendefinisikan subjek dengan menjelaskan identitas atau sifatnya",
         "jakarta adalah ibu kota indonesia merupakan definisi identitas kota"),

        ("kata termasuk mengelompokkan entitas ke dalam kategori yang lebih luas",
         "harimau termasuk karnivora berarti harimau berada dalam kelompok pemakan daging"),

        ("kata merupakan bentuk formal dari adalah yang menegaskan identitas atau klasifikasi",
         "batu merupakan benda mati menegaskan klasifikasi batu sebagai benda tidak hidup"),

        ("klasifikasi mengurutkan entitas ke dalam kelompok berdasarkan sifat-sifat bersama",
         "mangga dan apel diklasifikasikan sebagai buah karena punya biji dan daging buah"),

        ("kata yaitu memberikan penjelasan atau contoh spesifik dari pernyataan umum",
         "hewan mamalia yaitu kucing dan sapi memberikan contoh spesifik"),

        ("definisi menetapkan batas makna suatu konsep dengan menjelaskan ciri-ciri esensialnya",
         "segitiga didefinisikan sebagai bangun dengan tiga sisi yang menjelaskan ciri esensial"),

        ("kata ialah menegaskan definisi atau identitas dengan cara yang setara dengan adalah",
         "air ialah senyawa h2o menegaskan komposisi kimia air secara definisi"),

        ("kata disebut memberi nama atau label pada konsep yang baru diperkenalkan",
         "perubahan wujud dari cair ke gas disebut penguapan memberi nama pada proses"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 9: Kondisional
        # Jika...maka, apabila, ketika — hubungan
        # antara kondisi dan konsekuensi.
        # ═══════════════════════════════════════════════════════════════

        ("kata jika memulai kondisi yang harus dipenuhi agar konsekuensi terjadi",
         "jika hujan maka bawa payung berarti hujan adalah syarat membawa payung"),

        ("kata maka menghubungkan kondisi dengan konsekuensi logis yang mengikutinya",
         "jika rajin belajar maka nilai bagus menunjukkan konsekuensi dari usaha"),

        ("kata apabila merupakan variasi formal dari jika yang menandai kondisi",
         "apabila cuaca cerah kita pergi bermain berarti cuaca cerah adalah syaratnya"),

        ("kata ketika menandai titik waktu saat kondisi terpenuhi dan kejadian terjadi",
         "ketika matahari terbenam langit menjadi merah menunjukkan kejadian bersamaan"),

        ("kondisi jika tanpa maka tetap menyiratkan konsekuensi meskipun tidak eksplisit",
         "jika lapar makan berarti lapar menyebabkan tindakan makan"),

        ("kata kecuali jika menandai pengecualian dari kondisi umum yang telah dinyatakan",
         "semua hadir kecuali jika sakit berarti sakit adalah kondisi pengecualian"),

        ("kata tanpa kondisi berarti konsekuensi terjadi tidak peduli apapun keadaannya",
         "tanpa kondisi dia tetap datang berarti kehadirannya pasti terjadi"),

        ("kata asalkan menandai syarat minimal yang cukup untuk konsekuensi terjadi",
         "asalkan belajar akan lulus berarti belajar adalah syarat yang memadai"),

        # ═══════════════════════════════════════════════════════════════
        # Domain 10: Bilangan dan properti
        # Prima, genap, ganjil, kelipatan — sifat-sifat
        # bilangan dan klasifikasinya.
        # ═══════════════════════════════════════════════════════════════

        ("bilangan prima hanya bisa dibagi habis oleh satu dan dirinya sendiri",
         "7 adalah bilangan prima karena tidak bisa dibagi habis selain 1 dan 7"),

        ("bilangan genap adalah bilangan yang habis dibagi dua tanpa sisa",
         "8 adalah genap karena 8 dibagi 2 sama dengan 4 tanpa sisa"),

        ("bilangan ganjil adalah bilangan yang jika dibagi dua menghasilkan sisa satu",
         "5 adalah ganjil karena 5 dibagi 2 menyisakan 1"),

        ("bilangan kelipatan adalah hasil perkalian suatu bilangan dengan bilangan bulat",
         "12 adalah kelipatan 3 karena 3 kali 4 sama dengan 12"),

        ("faktor dari suatu bilangan adalah bilangan bulat yang membagi habis bilangan tersebut",
         "faktor dari 12 adalah 1 2 3 4 6 12 karena semuanya membagi habis 12"),

        ("bilangan komposit adalah bilangan yang memiliki faktor lebih dari dua",
         "6 adalah komposit karena faktornya 1 2 3 6 lebih dari dua faktor"),

        ("bilangan kuadrat adalah hasil perkalian suatu bilangan dengan dirinya sendiri",
         "9 adalah bilangan kuadrat karena 3 kali 3 sama dengan 9"),

        ("bilangan fibonacci diperoleh dengan menjumlahkan dua bilangan sebelumnya dalam deret",
         "8 adalah fibonacci karena 3 ditambah 5 sama dengan 8 dalam deret fibonacci"),
    ]

    return pairs
