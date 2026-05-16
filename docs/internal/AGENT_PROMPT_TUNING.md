# AGENT PROMPT — RSVS Tuning: Fix Auto-Induction Quality

## Context
Repo: SymbolicPuzzle3D
Binary target: `rsvs-realtest` (`backend/crates/rsvs-core/src/bin/rsvs-realtest.rs`)

Hasil test saat ini menunjukkan gap TRUE vs FALSE terlalu kecil (~5pp).
Root cause: parameter default terlalu ketat untuk short corpus (60 sentences).
Kita adalah yang menulis blueprint RSVS, jadi kita tahu ini bukan bug arsitektur —
ini tuning issue. Jangan ubah algoritma inti, hanya sesuaikan parameter.

---

## ROOT CAUSE ANALYSIS

### Problem 1: `entity_promote_n = 3` terlalu tinggi untuk test corpus
Sekarang: token harus muncul di ≥3 kalimat berbeda untuk dipromote.
Per domain hanya 15 kalimat → content words yang penting sering tidak mencapai threshold.
Fix: turunkan ke `entity_promote_n: 2`.

### Problem 2: `tau_overlap = 0.8` terlalu ketat
Sekarang: komposisi ditolak kalau overlap dengan known nodes < 80%.
Di awal corpus yang sparse, hampir semua komposisi ditolak.
Fix: turunkan ke `tau_overlap: 0.5`.

### Problem 3: `tau_compress = 0.3` membuang terlalu banyak komposisi
Komposisi dengan frekuensi komponen < 0.3 ditolak.
Di corpus kecil, hampir semua frekuensi rendah.
Fix: turunkan ke `tau_compress: 0.15`.

### Problem 4: `composition_min_confidence = 0.3` terlalu tinggi untuk early-stage graph
Fix: turunkan ke `composition_min_confidence: 0.15`.

### Problem 5: `min_cooc = 2` di AttentionConfig — pairs dengan cooc=1 diabaikan
Di corpus kecil, banyak pairs yang significant tapi hanya muncul 1x.
Fix: turunkan ke `min_cooc: 1`.

### Problem 6: `theta_assign = 0.30` — sense assignment terlalu ketat
Context sering tidak di-assign ke sense manapun → komposisi tidak terbentuk.
Fix: turunkan ke `theta_assign: 0.20`.

### Problem 7: `gamma_stopword = 0.70` — terlalu banyak token dikategorikan stopword
Token fungsional Indonesia (yang, di, dll.) punya global freq tinggi tapi perlu
dipertahankan karena mereka seeds. Tapi content words seperti "rumah", "sakit"
juga kena stopword filter padahal mereka penting.
Fix: naikkan ke `gamma_stopword: 0.85`.

---

## TASK — Edit `make_config()` di rsvs-realtest.rs

File: `backend/crates/rsvs-core/src/bin/rsvs-realtest.rs`

Ganti fungsi `make_config()` dengan versi berikut:

```rust
fn make_config() -> PipelineConfig {
    let custom_seeds: Vec<String> = vec![
        // Epistemological primitives (original 24)
        "exists".into(), "entity".into(), "relation".into(), "state".into(),
        "change".into(), "time".into(), "space".into(), "cause".into(),
        "effect".into(), "context".into(), "signal".into(), "pattern".into(),
        "memory".into(), "attention".into(), "value".into(), "agent".into(),
        "goal".into(), "risk".into(), "trust".into(), "identity".into(),
        "language".into(), "meaning".into(), "action".into(), "feedback".into(),
        // Indonesian functional words (grounding gate untuk teks Indonesia)
        "yang".into(), "di".into(), "dan".into(), "adalah".into(),
        "untuk".into(), "dengan".into(), "pada".into(), "dari".into(),
        "ke".into(), "itu".into(), "ini".into(), "tidak".into(),
        "sangat".into(), "setiap".into(), "sudah".into(), "seperti".into(),
        "sebuah".into(), "seorang".into(), "oleh".into(), "juga".into(),
        "atau".into(), "bisa".into(), "lebih".into(), "dalam".into(),
        "telah".into(), "akan".into(), "ada".into(), "banyak".into(),
    ];

    // Tuned SenseInductionConfig untuk short corpus
    let mut induction = rsvs::sense::SenseInductionConfig::default();
    induction.tau_overlap = 0.5;              // was 0.8 — terlalu ketat untuk sparse graph
    induction.tau_compress = 0.15;            // was 0.3 — buang terlalu banyak komposisi
    induction.composition_min_confidence = 0.15; // was 0.3 — terlalu tinggi early-stage

    // Tuned SenseConfig
    let mut sense = rsvs::sense::SenseConfig::default();
    sense.theta_assign = 0.20;               // was 0.30 — context lebih mudah di-assign ke sense
    sense.gamma_stopword = 0.85;             // was 0.70 — content words tidak kena filter
    sense.induction = induction;

    // Tuned AttentionConfig
    let mut attention = rsvs::attention::AttentionConfig::default();
    attention.min_cooc = 1;                  // was 2 — di corpus kecil, cooc=1 tetap penting

    PipelineConfig {
        entity_promote_n: 2,                 // was 3 — threshold lebih rendah untuk short corpus
        custom_seeds: Some(custom_seeds),
        sense,
        attention,
        tau_entity_learned: 0.10,            // was 0.15 — lebih mudah promote via learned score
        ..PipelineConfig::default()
    }
}
```

---

## TASK 2 — Tambah lebih banyak seed functional words Indonesia

Seed words tambahan di atas sudah mencakup: `oleh, juga, atau, bisa, lebih,
dalam, telah, akan, ada, banyak` — kata-kata yang sering muncul di teks
Indonesia tapi belum ada di seed list sebelumnya.

Ini BUKAN content words. Ini functional/grammatical words.
Content words (dokter, petani, komputer, dll.) tetap 100% auto-induced.

---

## TASK 3 — Tambah metric discriminability ke SUMMARY

Di bagian SUMMARY (`section("SUMMARY: ...")`) di rsvs-realtest.rs,
tambahkan metric baru setelah print TRUE/FALSE avg:

```rust
// Discriminability per domain
println!("\n  --- Discriminability per Domain ---");
let domains = vec![
    ("Budi(dokter)", r1.agree_pct, r2.agree_pct),
    ("Siti(petani)", r5.agree_pct, r4.agree_pct),
    ("Andi(guru)", c1.agree_pct, c2.agree_pct),
    ("Komputer", t1.agree_pct, t2.agree_pct),
    ("Sejarah", h1.agree_pct, h2.agree_pct),
    ("Gunung", r7.agree_pct, r6.agree_pct),
];

let mut all_pass = true;
for (name, t, f) in &domains {
    let gap = t - f;
    let status = if *gap > 0.0 { "✓ PASS" } else { "✗ FAIL" };
    if *gap <= 0.0 { all_pass = false; }
    println!("  {:20} TRUE={:.1}%  FALSE={:.1}%  gap={:+.1}pp  {}",
        name, t, f, gap, status);
}

println!("\n  Overall: {} / {} domains discriminable",
    domains.iter().filter(|(_, t, f)| t > f).count(),
    domains.len());

if all_pass {
    println!("  RESULT: ALL DOMAINS PASS — system discriminates TRUE from FALSE");
} else {
    println!("  RESULT: PARTIAL — some domains need more corpus data");
}
```

---

## TASK 4 — Build dan run, laporkan hasilnya

```bash
cd backend
cargo build --release --bin rsvs-realtest 2>&1 | grep -E "error|warning: unused" | head -20
./target/release/rsvs-realtest
```

Target yang ingin dicapai:
- Atoms promoted: ≥ 50 (sebelumnya 35)
- Compositions induced: ≥ 800 (sebelumnya 508)
- TRUE avg > FALSE avg dengan gap ≥ 10pp (sebelumnya ~5pp)
- ≥ 5/6 domains discriminable

---

## TASK 5 — Commit dan push

```bash
git add backend/crates/rsvs-core/src/bin/rsvs-realtest.rs
git commit -m "tune: lower thresholds for short-corpus auto-induction quality

- entity_promote_n: 3 → 2 (easier promotion)
- tau_overlap: 0.8 → 0.5 (accept more compositions early)
- tau_compress: 0.3 → 0.15 (keep more compositions)
- composition_min_confidence: 0.3 → 0.15
- min_cooc: 2 → 1 (count single co-occurrences)
- theta_assign: 0.30 → 0.20 (easier sense assignment)
- gamma_stopword: 0.70 → 0.85 (protect content words)
- tau_entity_learned: 0.15 → 0.10
- Add 10 more Indonesian functional seed words"
git push origin main
```

---

## CATATAN PENTING untuk agent

Jangan ubah:
- Algoritma appraise(), relate(), ingest_text() — tidak perlu disentuh
- Struct field names di PipelineConfig, SenseConfig, AttentionConfig
- appraise_against() — sudah benar

Yang boleh diubah:
- Hanya nilai parameter di make_config() di rsvs-realtest.rs
- Tambah seed words (functional words, bukan content words)
- Tambah metric di SUMMARY section

Filosofi: sistem RSVS sudah benar secara arsitektur.
Masalahnya adalah default parameter dirancang untuk corpus besar (ribuan kalimat).
Test kita pakai 60-90 kalimat → perlu parameter yang lebih relaxed.
