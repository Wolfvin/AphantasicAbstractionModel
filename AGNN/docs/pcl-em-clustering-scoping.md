# Scoping: EM-style Iterative Clustering untuk PCL (Round 42)

**Status**: SCOPING SAJA — belum ada kode ditulis. Mengikuti disiplin
Round 11/19 (scoping lengkap sebelum baris kode pertama).

## 1. Masalah yang mau diselesaikan

Dibuktikan empiris Round 41: `_cluster_action_group_qkv`/`_cluster_particles`
itu **greedy sequential, satu arah**. Begitu 2 token (A, B) ke-cluster
TERPISAH dalam satu pass (karena diproses lebih dulu, freq lebih tinggi),
token C yang datang BELAKANGAN — walau secara objektif "menjembatani"
A dan B sama rata — cuma bisa **gabung ke SATU** cluster (lewat argmax),
TIDAK PERNAH menyebabkan A dan B digabung ulang.

Target: cluster lama harus bisa **direvisi** (gabung/pecah ulang) kalau
data baru menunjukkan partisi yang lebih koheren secara global —
bukan cuma "tempel di akhir urutan".

## 2. Definisi konvergensi (HARUS jelas sebelum coding)

EM-style klasik: `assign → update centroid → re-assign SEMUA token →
ulang sampai TIDAK ADA assignment yang berubah (converged) ATAU max
iterasi tercapai`.

**Pertanyaan yang harus dijawab dulu**:
- **Apa yang di-re-assign?** Action token (981+ di exploration), atau
  particle (331+)? Atau keduanya, terpisah?
- **Centroid di-update gimana?** Mean dari SEMUA member SETELAH setiap
  iterasi (drift), ATAU tetap seed-anchored (anti-chain-merge fix PR
  #106/107) tapi seed-nya bisa BERGANTI per iterasi kalau member baru
  lebih representatif?
  - **RISIKO**: kalau centroid jadi mean-drift (bukan seed-anchored),
    kita BALIK ke bug PR #106 (chain-merge) yang sudah diperbaiki
    susah-susah jadi seed-anchored. EM-style dan anti-chain-merge bisa
    BERTENTANGAN secara desain — ini harus diselesaikan SEBELUM coding,
    bukan ditemukan saat testing.
- **Kriteria stop**: "tidak ada assignment berubah" — tapi kalau ada
  OSCILLATION (token X loncat antara cluster 1↔2 tiap iterasi, tidak
  pernah stabil)? Perlu max-iteration cap + deteksi oscillation
  eksplisit (BUKAN infinite loop diam-diam).

## 3. Dampak ke SETIAP caller existing (audit wajib sebelum coding)

`cluster_id_of`/`action_clusters`/`particle_clusters` dibaca oleh:

| Caller | Risiko kalau cluster_id bergeser LEBIH SERING |
|---|---|
| `_is_action_token` | Bergantung `cluster_id_of[token] >= 0` — kalau token loncat cluster per iterasi EM, hasil tag_sentence/spo bisa tidak stabil DALAM SATU train() call (bukan cuma antar versi). |
| `spo()`/`spo_embedded()`/`tag_sentence()` | Semua baca `is_action_token`/`is_particle_token` — efek domino dari di atas. |
| `mark_clause_coordinator_clusters()` | API ini terima `Set[int]` cluster_id — kalau koordinator (`dan`/`atau`) loncat cluster_id antar iterasi EM, caller (bootstrap_classifier.py, explore_clusters.py) HARUS panggil ulang SETELAH konvergen, bukan di tengah iterasi. |
| `label_clusters()`/`label_particle_clusters()` | SAMA — harus dipanggil SETELAH konvergen final, bukan per-iterasi. |
| `get_relation_type_for_action()` | Sudah match-by-content (bukan akses cluster_id langsung) — AMAN by design (keputusan lama, lihat tabel Keputusan Desain). |
| `inspect_cluster_details()`/`inspect_particle_clusters()` | Cuma display, aman. |
| `chat_agnn.py`'s `:trace` | Replay scoring SEKALI — kalau EM jalan multi-iterasi, `:trace` perlu update buat tunjukkan SEMUA iterasi, bukan cuma 1 snapshot. |

**Kesimpulan audit**: SEBAGIAN BESAR caller sudah aman (pola
match-by-content sudah jadi keputusan desain lama, untungnya). Yang
paling rawan: `mark_clause_coordinator_clusters`/`label_*` HARUS
dipastikan dipanggil SETELAH konvergensi final, tidak boleh di
tengah iterasi.

## 4. Rencana validasi sebelum-sesudah (wajib, per pelajaran Round 36/40)

1. **Eksperimen terkontrol Round 41 (membaca/memasak/menggambar)** jadi
   test case PERTAMA — pastikan EM-style BERHASIL menggabung
   `membaca`+`memasak` lewat `menggambar` sebagai jembatan (hasil yang
   GAGAL di versi greedy sekarang).
2. **Regression test lama wajib tetap hijau**: terutama
   `test_particle_cluster_purity_no_chain_merge_regression` (cluster
   `tidak` harus tetap ≤6 anggota) — EM-style TIDAK BOLEH membuka
   kembali chain-merge yang sudah diperbaiki PR #105-107.
3. **464 test full suite** harus tetap hijau SETELAH ganti algoritma —
   sama kontrak dengan Round 25 (Q/K/V-ify Brown clustering).
4. **Ablation compute cost**: EM-style genuinely lebih mahal (multi-
   iterasi vs 1 pass) — profiling WAJIB (pelajaran Round 24: jangan
   optimasi/ganti algoritma tanpa data, di sini arahnya kebalik — jangan
   PERLAMBAT tanpa data juga). Ukur n-iterasi rata-rata sampai
   konvergen di corpus production DAN exploration (skala beda jauh).
5. **State file regenerate** (kontrak issue #92) — WAJIB karena
   algoritma berubah signifikan, sama seperti Round 25.

## 5. Keputusan desain yang BELUM diambil (perlu dijawab sebelum scoping selesai)

1. EM diterapkan ke **action clustering SAJA**, **particle clustering
   SAJA**, atau **keduanya**? (Rekomendasi: action dulu — particle
   sudah punya isu sendiri yang lebih genting, Round 40, jangan
   tumpuk 2 perubahan besar di komponen yang sama sekaligus.)
2. Centroid mean-drift vs seed-anchored-tapi-bisa-ganti-per-iterasi —
   **ini keputusan paling kritis**, perlu didiskusikan dengan user
   SEBELUM baris kode pertama, karena langsung bersinggungan dengan
   fix anti-chain-merge yang sudah established.
3. Threshold (`qkv_action_similarity_threshold`) tetap sama, atau
   perlu re-kalibrasi untuk rezim iteratif? (Pelajaran Round 40:
   ganti cara hitung SELALU butuh re-kalibrasi threshold, jangan
   asumsikan angka lama masih berlaku.)

## 6. Rekomendasi langkah selanjutnya

**JANGAN langsung coding.** Urutan yang disarankan:
1. Jawab 3 keputusan terbuka di atas (kalau perlu, AskUserQuestion ke
   user — terutama soal centroid mean-drift vs seed-anchored).
2. Prototype TERISOLASI (bukan ganti `_cluster_action_group_qkv`
   langsung) — fungsi baru `_cluster_action_group_em()` paralel,
   dites di skenario Round 41 dulu sebelum dipasang ke pipeline asli.
3. Baru setelah prototype lolos test #1-3 di section 4 — pasang ke
   pipeline asli, validasi penuh, BARU hapus versi greedy lama.
