# AGNN — Katalog Cluster Lengkap (Exploration Sandbox)

**Snapshot setelah**: commit `06b2670` (54 batch feed, 6711 token, 944 action, 295 action cluster, 18 particle cluster). Training dihentikan setelah rasio singleton plateau 3 batch berturut-turut (~66%, lihat `project-self-ai.md` Round 36-37).

**Cara baca dokumen ini**:
- Setiap cluster dibentuk PURE STATISTIK (Q/K/V cosine similarity di atas distribusi objek/konteks) — **label di bawah ini ditulis SETELAH cluster terbentuk** (cluster-first, label-second — prinsip non-negotiable AGNN). Label adalah interpretasi manusia, BUKAN input ke proses cluster.
- **Size bukan indikator noise.** Cluster 2-3 anggota bisa lebih presisi maknanya daripada cluster besar (lihat Cluster 230 di bawah — 2-3 anggota, paling jelas maknanya).
- **Singleton (196 cluster)** belum punya pembanding sama sekali — ditandai `PENDING-DEPTH`, bukan "noise". Perlu lebih banyak kalimat berisi token itu sebelum bisa dinilai benar/salah.
- `top_objects` = objek yang paling sering muncul bersama anggota cluster ini (digabung dari `action_object_freq` semua anggota).

---

## 1. Particle Clusters (18)

Particle cluster dibentuk dari token yang sudah dikecualikan dari slot ACTION (function word / connector / modifier), di-cluster lagi berdasar fitur posisi 4-dimensi (bukan distribusi objek seperti action cluster).

| ID | Size | Anggota (contoh) | Label interpretasi |
|---|---|---|---|
| 0 | 111 | `dan, atau, oleh, ke, dari, untuk, dengan, kepada, ...` + angka (`1,10,100,...`) + noun leak (`famili,jawa,planet,pulau`) | **CAMPUR BESAR — beberapa sub-kategori grammar tergabung jadi 1** (preposisi + konjungsi + numeral + noise). Ini bukti konkret temuan Round 19-21 "cluster besar makin kotor" — butuh dipecah, BELUM dieksekusi. |
| 8 | 74 | `maka, makanya, sehingga, tetapi, namun, dianggap, dikenal, ditemukan, ...` | **CAMPUR — konektor diskursus (maka/sehingga/tetapi) + verb pasif leak (dianggap/dikenal)**. Sama kelas masalah dengan #0, skala lebih kecil. |
| 1 | 19 | `ada, adanya, bentuk, jumlah, periode, sekitar, tingkat, pada` | Kuantifier/eksistensial — cukup koheren ("ada", "sekitar", "jumlah" semua menunjuk kuantitas/eksistensi). |
| 12 | 17 | `aroma, foton, kekebalan, kekuasaan, vitamin, zat, versi` | **Noun leak** — semua ini content noun, bukan particle genuine. Particle clustering keliru menangkap noun langka. |
| 4 | 17 | `agar, karena, sejak, selama, walaupun, secara, semua` | **SUBORDINATOR — paling koheren di antara semua particle cluster.** `agar/karena/walaupun` genuine konjungsi subordinatif. Ini cluster yang paling siap dipakai sebagai dasar grammar-class "SUBORDINATING_CONJUNCTION". |
| 3 | 13 | `bukan, sangat, merupakan, diklasifikasikan, sehingga, pasti` | Campur modal/intensifier (`sangat,pasti`) + copula leak (`merupakan`). |
| 9 | 7 | `akibatnya, lantaran, mana, peran, jawab` | Konektor kausal sekunder (`akibatnya,lantaran` = "akibatnya/karena itu") — tema cukup jelas. |
| 10 | 4 | `efektif, hampir, orbit, peningkatan` | Lemah, tidak ada tema jelas — kandidat `PENDING-DEPTH` walau bukan singleton. |
| 2 | 3 | `adalah, bukanlah, tampak` | **Copula negatif/positif** — koheren (semua predikat nominal kopula). |
| 5 | 2 | `golongan, kategori` | Noun sinonim — bocor dari klasifikasi, bukan particle genuine. |
| 6 | 2 | `bahwa, peranan` | `bahwa` (komplementizer, baru stabil sejak corpus depth Round 26) numpang ke `peranan` (noun) — masih coincidence statistik, BELUM murni. |
| 7 | 2 | `bisa, cenderung` | Modal — koheren. |
| 11,13,14,15,16,17 | 1 masing² | `hubungan, impor, metamorfosis, tahan, beliung, pelemakan` | `PENDING-DEPTH`. |

---

## 2. Action Clusters — size ≥ 4 (27 cluster, paling matang)

| ID | Size | Anggota | top_objects | Label |
|---|---|---|---|---|
| 0 | 20 | `berkategori, klasifikasi, memeriksa, menangkap, mencatat, mengukur, menunjukkan, pemakan, tergolong, termasuk, terungkap, ...` | mamalia, tumbuhan, reptil, ikan, serangga, logam, aves, pisces | **CATEGORICAL** — verb klasifikasi/taksonomi. Mirror persis RelationType CATEGORICAL di production. |
| 5 | 19 | `berlandaskan, berperan, butuh, memanfaatkan, membutuhkan, memerlukan, memproduksi, menyimpan, perlu, tergantung, ...` | air, oksigen, makanan, udara, bensin, listrik, modal, darah | **FUNCTIONAL** — verb kebutuhan/pemanfaatan resource. Mirror RelationType FUNCTIONAL. |
| 104 | 18 | `anorganik, bergoyang, eksekutif, ketahanan, kokoh, legislatif, licin, mengeras, nabati, segar, serealia, terbentang, ...` | `""` (sentinel kosong, 58x) | **PREDICATE-ADJECTIVE** — kata sifat predikatif tanpa objek (mekanisme Round 9: "rumah itu kokoh/licin/segar"). Sentinel `""` itu sendiri bukti mekanisme bekerja persis seperti didesain. |
| 4 | 15 | `berakibat, berkontribusi, membuat, memengaruhi, memicu, mengakibatkan, menghasilkan, menghindari, menyebabkan, ...` | banjir, kekeringan, longsor, penyakit, ledakan, diabetes, keracunan, asap | **CAUSAL** — mirror RelationType CAUSAL persis. |
| 10 | 14 | `bertahan, berusaha, kemudian, ketika, lalu, mengandalkan, saat, sebelum, setelah, ...` | hujan, cair, dingin, panas, musim, pagi, kering, gelap | **TEMPORAL** — mirror RelationType TEMPORAL (subordinator waktu + state-change cuaca). |
| 3 | 13 | `biaya, keliru, lahan, memantau, mempelajari, menggali, menurut, permukaan, sebab, terjadi, ...` | tanah, air, lainnya, tiba, pertanian, tinggi | **CAMPUR — verb investigasi (memantau/mempelajari/menggali) + noise posisi.** `permukaan` di sini adalah wall yang sudah dikonfirmasi Round 27-28 (compound-noun coincidence, BELUM ada fix). |
| 209 | 11 | `beragam, berbahaya, berbeda, berlawanan, membedakan, menegaskan, terbagi, terhitung, ...` | mamalia, reptil, logam, tumbuhan, ikan, serangga, aves, pisces | **DIFFERENTIAL** — mirror RelationType DIFFERENTIAL. Objek SAMA PERSIS dengan Cluster 0 (CATEGORICAL) — wajar, sama-sama dari konteks taksonomi biologi. |
| 2 | 10 | `akhir, memenuhi, menanam, mendekati, mengancam, menghuni, tenaga, tropika, ...` | perdagangan, afrika, crash, hari, laut, air | Tema lemah: verb habitat (`menghuni/mendekati/mengancam`) + noise. |
| 1 | 9 | `berburu, berdampak, makan, memijat, memiliki, mengarahkan, mengganggu, ...` | tubuh, persen, masyarakat, tulang, gigi, manusia | Campur — `berburu/makan` (predasi) + verb generik, heterogen. |
| 210 | 8 | `bergantung, bersandar, bertumpu, ditebang, organ, terbuka, wajib` | air, oksigen, makanan, udara, bensin, modal, darah, listrik | **FUNCTIONAL (varian)** — objek SAMA dengan Cluster 5 tapi tidak ter-merge (tie-break threshold). "Bergantung pada resource" — semantik identik dengan Cluster 5, kandidat MERGE manual kalau threshold dinaikkan. |
| 6 | 8 | `berukuran, cara, mengklasifikasikan, menyangkut, menyerap, terang, ...` | matahari, rakyat, ekonomi, asia, panas, risiko | Campur, tema lemah. |
| 7 | 8 | `berolahraga, menawarkan, mendengarkan, menempati, menjadi, ...` | jantung, lembut, segar, produk, populer | Campur, tidak ada tema jelas. |
| 19 | 7 | `begitu, berbuah, berlebih, disiram, ditambah, kebutuhan, menyiram` | turun, asin, layu, manis, naik, otot | **PERAWATAN TANAMAN** — `disiram/menyiram` + predikat-adjektiva kondisi tanaman (`layu/asin/manis`) — tema koheren genuine. |
| 14 | 6 | `berwarna, budidaya, diperkirakan, mendukung, menerapkan, pertama` | lalu, awal, kambrium, inggris, prancis, hitam | Tema historis-temporal lemah. |
| 213 | 6 | `berfungsi, bermanfaat, bertanggung, hubungannya, memainkan, tertentu` | tubuh, masyarakat, sosial, negara, barang | **FUNGSI/MANFAAT** — tema koheren (semua tentang fungsi/peran/manfaat sesuatu). |
| 9 | 6 | `bertulang, besar, mengejar, organisme, terakhir, tertinggi` | responsif, asia, kipas, tsunami, bumi, neuron | Campur, tidak koheren. |
| 93 | 6 | `dipangkas, dipindahkan, keuangan, mencair, mengundang, terinfeksi` | naik, meluap, sungai, banjir, warga | Tema lemah: perubahan-keadaan (mencair/naik/meluap). |
| 241 | 5 | `bagi, berkumpul, celah, mendadak, relatif` | bumi, liar, awan, sinapsis, normal | Campur, tidak koheren. |
| 15 | 4 | `cukup, memang, sebenarnya, terlihat` | segar, lembut, kuat, mulia, hangat, lentur, manis | Predikat-adjektiva + diskursus-modal — varian kecil dari Cluster 104. |
| 162 | 4 | `berbahasa, mempertahankan, mengelola, terancam` | indonesia, selatan, pasar, migrasi, modern | Tema lemah: governance/manajemen. |
| 220 | 4 | `dua, mencadangkan, muara, temperatur` | tahun, induknya, industri, jalan, bulan | Campur, tidak koheren. |
| 232 | 4 | `120, memegang, mempengaruhi, mengacu` | manusia, erosi, dunia, lereng, perkotaan | Campur, tidak koheren. |
| 25 | 4 | `memunculkan, mendatangkan, menimbulkan, meningkat` | panda, terbentuk, naik, inflasi, pusing, terbakar | **CAUSAL (varian)** — bersaudara dengan Cluster 4, terpisah karena connector-split (Round 19's `action_connector_signature`). |
| 40 | 4 | `berkembang, bertindak, menganggap, menyediakan` | bawaan, orang, elektronik, eropa | Campur, tidak koheren. |
| 63 | 4 | `bermuatan, menembus, menentukan, mengenai` | terjadi, positif, terurai, elektromagnetik | Tema lemah: fisika (`bermuatan/elektromagnetik/positif`). |
| 8 | 4 | `antara, dalam, juta, mendefinisikan` | petir, aktif, perusahaan, ursidae | Campur + particle leak (`antara,dalam`). |

---

## 3. Action Clusters — size 2-3 (76 cluster)

Yang **PALING PRESISI maknanya** (highlight, jangan dilewatkan):

| ID | Anggota | top_objects | Kenapa istimewa |
|---|---|---|---|
| **230** | `diperbaiki, diperiksa, diselidiki` | teknisi, tukang, petugas, polisi | **Cluster paling presisi di seluruh katalog.** Semua "objek" sebenarnya AGEN pasif (siapa yang memperbaiki/memeriksa/menyelidiki) — profesi reparasi & investigasi. Confirmed Round 21 sebagai temuan genuine, bukan bug (walau secara teknis "objek" yang ditangkap itu agen pasif — root cause `tokens[-1]` blind extraction yang sudah didokumentasikan, TAPI hasil cluster-nya tetap koheren karena posisi agen pasif itu rigid). |
| 111 | `diteteskan, nipis` | asam (8x) | **FRAGMEN NOISE terkonfirmasi** — ini "asam nipis" (jeruk nipis) yang salah ke-split jadi 2 token action. Sama kelas dengan Cluster 96 (`berpendapat`+`high-end`) yang sudah didokumentasikan Round 22. |
| 101 | `clic, salah` | terkunci (7x) | Kemungkinan fragmen teknis ("klik" terpotong) — noise. |
| 42 | `goreng, memasak, membaca` | bawang, nasi, sup, buku | 2 sub-tema tergabung kebetulan (memasak+goreng → bawang/nasi/sup; membaca → buku) — bukan 1 kategori genuine, cuma kebetulan threshold. |

Sisanya (72 cluster) — tema bervariasi dari campuran genuine-tapi-tipis sampai kebetulan statistik murni:

`berdarah-mempelajari` (lihat di atas), `mencapai/mengelilingi/terbesar`→astronomi (kilometer,dunia,surya,bumi), `berasal/bersayap`→evolusi hewan (beruang,sapi,terbang), `dicuci/dimasak`→pekerjaan rumah, `mendapatkan/menerima`→transaksi (dana), `mencetak/menendang`→olahraga (gol,lawan), `menyimpulkan/percaya`→verb kognitif (berhasil), dan ~60 cluster lain dengan tema lemah/campur — daftar lengkap ada di `_cluster_dump.json` (working tree, belum di-commit, tersedia untuk inspeksi lanjutan kalau dibutuhkan).

---

## 4. Singleton — 196 token, `PENDING-DEPTH`

Belum punya pembanding sama sekali — TIDAK dinilai benar/salah, hanya menunggu lebih banyak kalimat:

```
a, abad, angkatan, bagian, bahan, bahasa, bandang, bekerja, bela, belajar, belakang,
berapi, berarti, berbagai, berbau, berbentuk, berdasarkan, berdengung, berevolusi,
berkaitan, berkekuatan, berkepanjangan, berkicau, berkontraksi, berlaku, berlalu,
berlangsung, bermacam-macam, bernama, berpendapat, bersaksi, bersifat, bersorak,
bertujuan, beruang, berupa, bervariasi, diabetik, dibekukan, dibuka, dicetak,
didiamkan, didirikan, dikatakan, dikembangkan, dilindungi, diperlukan, disedu,
ditulis, festival, higgs, high-end, itu, jalar, jenner, kaktus, kondisi, konservasi,
kue, lapuk, lawas, lebih, longgar, makhluk, meluruh, memakan, memaksa, memancarkan,
memanen, memasang, membagi, membagikan, membahayakan, membawa, membeli, membentuk,
memberi, memberikan, membuka, memfasilitasi, memperbaiki, mempercepat, mempererat,
memperkirakan, mempunyai, memulai, memungkinkan, memutar, menabung, menampilkan,
menandai, menangani, menarik, mencampur, mencari, menciptakan, mencoba,
mencurigakan, mendekat, mendesak, menemukan, menetapkan, mengajari, mengaku,
mengamati, menganalisis, mengantar, mengatur, mengenali, mengganti, menggunakan,
mengguyur, menghubungkan, mengikuti, mengkalibrasi, mengurangi, meningkatkan,
meningkatnya, menjadikan, menjalani, menjalankan, menjelaskan, menulis, menumpuk,
menurun, menutup, menyala, menyalakan, menyampaikan, menyelenggarakan, merujuk,
milik, mustahil, neural, oven, paling, pemanfaatan, pembentukan, pemutusan,
pengendalian, penggunaan, perangkat, perdebatan, phishing, predator, purba, rasa,
rata-rata, raya, rekreasi, serikat, set, spesies, stellers, suatu, sudah, swasta,
tempat, terasa, terbang, terbentuk, terbit, tercemar, tercepat, tercipta,
terdaftar, terdapat, terdekat, terdiri, tereksitasi, terganggu, terhadap,
terkandung, terkenal, terkendali, terkunci, terlaksana, terlepas, terletak,
terluka, terputus-putus, tersangkut, tersebut, tersulit, tersusun, tertua, terus,
tiga, toksoid, tunai, umumnya, unggul, usaha, usb-c, utama, uv
```

**Catatan khusus**: `berapi` di daftar ini — ini Round 33's temuan morphology false-positive (`ber-`+`api`, "gunung berapi" = compound noun "volcano", bukan verb "ber-api"). Singleton-nya BUKAN karena kurang data semata — ada gap morfologi terpisah yang sudah didokumentasikan, BUTUH supervisi sebelum fix.

---

## Ringkasan Kuantitatif

| Kategori | Jumlah | % |
|---|---|---|
| Action cluster size ≥4 (matang, siap label formal) | 27 | 9% |
| Action cluster size 2-3 (sebagian presisi, sebagian noise) | 76 | 26% |
| Action cluster singleton (`PENDING-DEPTH`) | 196 | 66% |
| Particle cluster (semua size) | 18 | — |

**5 dari 27 cluster size≥4 langsung mirror RelationType production** (CATEGORICAL, FUNCTIONAL, CAUSAL, TEMPORAL, DIFFERENTIAL) — bukti AGNN convergent ke kategori yang SAMA dari corpus yang BEDA total (production curated vs exploration Wikipedia mentah). Ini validasi independen kuat untuk arsitektur zero-bias-nya.
