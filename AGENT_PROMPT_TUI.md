# RSVS Agent Prompt — TUI + Contextual Appraise Mode

## Context

Repo: SymbolicPuzzle3D
Crate: `backend/crates/rsvs-core` (lib name: `rsvs`, crate-type: `["cdylib", "rlib"]`)
Public API entry point: `rsvs::pipeline::Rsvs`
Key types: `Rsvs`, `AppraiseResult`, `IngestStats`, `RelateResult`, `PipelineConfig`
Existing binary: `backend/crates/rsvs-core/src/bin/rsvs-smoke.rs` (gunakan sebagai referensi)

---

## TASK 1 — Tambah `appraise_against` method ke `Rsvs`

File: `backend/crates/rsvs-core/src/pipeline/modes.rs`

Tambahkan method baru di bawah `appraise()`:

```rust
/// Appraise `statement` hanya berdasarkan `context` — bukan seluruh graph.
/// Context di-ingest ke instance temporary yang isolated, lalu statement di-appraise
/// terhadap instance itu saja. Graph utama tidak berubah.
pub fn appraise_against(&self, context: &str, statement: &str) -> AppraiseResult {
    // 1. Buat Rsvs instance temporary dengan config yang sama
    let mut temp = Rsvs::new(self.config.clone()).expect("temp rsvs");

    // 2. Ingest context ke instance temporary
    let _ = temp.ingest_text(context);

    // 3. Appraise statement terhadap temp instance
    temp.appraise(statement)
}
```

Tambahkan juga ke public exports di `backend/crates/rsvs-core/src/pipeline/mod.rs`
dan expose di `backend/crates/rsvs-core/src/lib.rs` jika belum.

---

## TASK 2 — Buat TUI binary baru

Buat file: `backend/crates/rsvs-core/src/bin/rsvs-tui.rs`

Tambahkan ke `backend/crates/rsvs-core/Cargo.toml`:
```toml
[[bin]]
name = "rsvs-tui"
path = "src/bin/rsvs-tui.rs"

[dependencies]
ratatui = "0.29"
crossterm = "0.28"
```

### TUI Layout (terminal full-screen, ratatui)

```
┌─────────────────────────────────────────────────────────┐
│  RSVS v8.3 — Recursive Symbolic Vector Space      [TUI] │
├──────────────────┬──────────────────────────────────────┤
│  GRAPH STATUS    │  OUTPUT                              │
│  atoms: 0        │                                      │
│  edges: 0        │                                      │
│  mode: NORMAL    │                                      │
│                  │                                      │
├──────────────────┴──────────────────────────────────────┤
│  > _                                                    │
├─────────────────────────────────────────────────────────┤
│  [I]ngest [A]ppraise [C]ontext [R]elate [Q]uit [?]help │
└─────────────────────────────────────────────────────────┘
```

- Panel kiri: status graph (atom count, edge count, mode aktif)
- Panel kanan: output hasil command terakhir, scrollable
- Bar bawah: input field aktif
- Footer: keyboard shortcuts

### Mode TUI

**NORMAL mode** — navigasi dengan keyboard shortcut:
- `i` → masuk INSERT mode, ingest text
- `a` → masuk APPRAISE mode, appraise terhadap seluruh graph
- `c` → masuk CONTEXT mode (dua langkah: input cerita dulu, lalu statement)
- `r` → masuk RELATE mode, input concept
- `q` → quit
- `?` → toggle help overlay

**INSERT mode** (setelah tekan `i`):
- User mengetik teks bebas (bisa multi-line, Enter untuk submit)
- Kirim ke `rsvs.ingest_text(&input)`
- Tampilkan hasil: atoms promoted, tokens processed
- Kembali ke NORMAL mode

**APPRAISE mode** (setelah tekan `a`):
- User ketik statement
- Kirim ke `rsvs.appraise(&statement)`
- Tampilkan:
  ```
  Appraise: <verdict> (<agree>% agree / <disagree>% disagree)
  Support  : token1 (0.92) | token2 (0.87)
  Conflict : token3 (0.12)
  ```

**CONTEXT mode** (setelah tekan `c`) — INI YANG BARU:
- Step 1: prompt "Masukkan cerita/konteks:" → user input paragraf bebas
- Step 2: prompt "Masukkan statement untuk diuji:" → user input statement
- Kirim ke `rsvs.appraise_against(&context, &statement)`
- Tampilkan output sama seperti APPRAISE mode
- Label output: "Contextual Appraise (isolated)"
- Tegaskan: graph utama TIDAK berubah

**RELATE mode** (setelah tekan `r`):
- User ketik concept
- Kirim ke `rsvs.relate(&concept)`
- Tampilkan related nodes dengan edge count

### Warna (ratatui Style)

Gunakan warna yang konsisten:
- Verdict `agree` → Green
- Verdict `disagree` / `conflict` → Red  
- Verdict `mixed` → Yellow
- Verdict `novel` → Cyan
- Label/header → Bold White
- Score tinggi (>0.7) → Green, medium (0.4–0.7) → Yellow, rendah (<0.4) → Red
- Mode aktif di footer → Bold Cyan (highlight)

### Error handling

Jika `ingest_text` atau `appraise` return error/empty:
- Tampilkan pesan error di output panel dengan warna Red
- Jangan crash — kembali ke NORMAL mode

---

## TASK 3 — Update workspace Cargo.toml

File: `backend/Cargo.toml`

Pastikan `ratatui` dan `crossterm` hanya jadi dependency untuk binary `rsvs-tui`,
bukan untuk lib (gunakan `[target.'cfg(...)'.dependencies]` atau cukup taruh
di `Cargo.toml` crate-level dengan feature gate jika perlu).

Paling simpel: taruh langsung di `backend/crates/rsvs-core/Cargo.toml`
di bawah `[dependencies]` — ratatui dan crossterm hanya dipakai oleh binary,
tidak akan masuk ke cdylib.

---

## TASK 4 — Update `backend/Cargo.toml` workspace members

Pastikan tidak ada yang perlu diubah — binary `rsvs-tui` sudah masuk otomatis
karena berada di dalam crate `rsvs` yang sudah jadi workspace member.

Build command setelah selesai:
```bash
cd backend
cargo build --release --bin rsvs-tui
```

Binary tersedia di: `backend/target/release/rsvs-tui`

---

## TASK 5 — Demo scenario untuk buktikan ke skeptic

Setelah TUI jalan, test dengan skenario ini:

### Skenario: Cerita vs Statement Bertolakbelakang

**Context (cerita):**
```
Budi adalah seorang dokter yang bekerja di rumah sakit. Setiap hari Budi 
menyembuhkan pasien dengan memberikan obat-obatan. Rumah sakit tempat Budi 
bekerja sangat besar dan modern. Budi sangat dihormati karena keahliannya.
```

**Statement 1 (benar):**
```
Budi bekerja di rumah sakit sebagai dokter.
```
Expected: `agree` tinggi

**Statement 2 (bertolakbelakang):**
```
Budi adalah seorang petani yang menanam padi di sawah.
```
Expected: `disagree` atau `conflict` — karena "petani" dan "sawah" tidak ada 
dalam context, dan "dokter" + "rumah sakit" ada di context.

**Statement 3 (partial/ambigu):**
```
Budi bekerja membantu orang lain setiap hari.
```
Expected: `mixed` atau `agree` — secara semantik konsisten tapi tidak 
semua token ada di context.

### Ini yang membuktikan ke skeptic:
- Tidak ada compose manual sama sekali
- Atom di-induce dari teks bebas (cerita Budi)
- Contradiction detection tanpa pre-defined ontology
- Isolated context → graph utama tidak terkontaminasi

---

## Expected file changes summary:
1. `backend/crates/rsvs-core/src/pipeline/modes.rs` — tambah `appraise_against()`
2. `backend/crates/rsvs-core/src/bin/rsvs-tui.rs` — **NEW** TUI binary
3. `backend/crates/rsvs-core/Cargo.toml` — tambah ratatui + crossterm dependency + [[bin]] entry
