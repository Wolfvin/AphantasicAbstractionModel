# AGENT PROMPT — Losion Bridge: Neuro-Symbolic Verification + Dual Memory untuk RSVS

## Context & Philosophy

Kita adalah yang menulis blueprint RSVS (Recursive Symbolic Vector Space).
Kita juga menulis Losion (Hybrid AI Framework dengan Tri-Jalur Architecture).

Sekarang kita ambil dua konsep dari Losion dan adaptasi ke RSVS:

1. **Neuro-Symbolic Verification** → `appraise()` jadi dua layer: graph-symbolic scoring
   (yang sudah ada) + structural verification pass yang explain *mengapa* verdict itu keluar
2. **Dual Memory** → RSVS punya dua layer graph: Working Graph (session, volatile) +
   Long-term Graph (persistent, consolidated). `appraise_against()` yang sudah ada
   adalah prototype Working Graph — sekarang kita formalisasi.

Source files dari Losion ada di folder `losion_bridge/` sebagai referensi arsitektur:
- `losion_bridge/neuro_symbolic_src.py` — referensi VerificationStatus, feedback loop
- `losion_bridge/dual_memory_src.py` — referensi WorkingMemory ring buffer pattern
- `losion_bridge/reflection_src.py` — referensi Reflection + lesson storage pattern

**PENTING**: Jangan port Python ke Rust 1:1. Ambil *konsep* dan *pattern*-nya,
implementasikan dalam idiom Rust yang sesuai dengan arsitektur RSVS yang sudah ada.
Tidak ada PyTorch, tidak ada neural network — RSVS adalah symbolic system.

---

## TASK 1 — `AppraiseVerdict` struct + verification pass di `modes.rs`

File: `backend/crates/rsvs-core/src/pipeline/modes.rs`

### 1a. Tambah struct `AppraiseVerdict`

Setelah `AppraiseResult` struct (atau di `types.rs`), tambahkan:

```rust
/// Detailed verdict dari appraise — menjelaskan *mengapa* verdict keluar.
/// Terinspirasi dari Losion's VerificationStatus + VerificationResult.
/// Di Losion: neural verifier menghasilkan confidence + error_type + feedback.
/// Di RSVS: symbolic verifier menghasilkan token-level explanation dari graph.
#[derive(Debug, Clone)]
pub struct AppraiseVerdict {
    /// Verdict ringkas: "agree" | "disagree" | "mixed" | "novel"
    pub verdict: String,
    /// Agree percentage (0–100)
    pub agree_pct: f32,
    /// Disagree percentage (0–100)
    pub disagree_pct: f32,
    /// Token-level support evidence: (token, score, reason)
    /// reason: "structural" | "convergent" | "seed" | "novel"
    pub support: Vec<(String, f32, String)>,
    /// Token-level conflict evidence: (token, score, reason)
    pub conflict: Vec<(String, f32, String)>,
    /// Human-readable explanation dari verdict
    /// Contoh: "3 tokens structurally grounded (budi, dokter, rumah).
    ///          1 token conflicts via absent composition (petani)."
    pub explanation: String,
    /// Apakah ini contextual appraise (isolated) atau global
    pub is_contextual: bool,
    /// Gap antara agree dan disagree — makin tinggi makin confident
    pub confidence_gap: f32,
}
```

### 1b. Tambah method `appraise_verbose()` di `impl Rsvs`

Method ini mengembalikan `AppraiseVerdict` yang lebih kaya dari `appraise()`.
`appraise()` tetap ada (backward compatible) — `appraise_verbose()` adalah wrapper.

```rust
pub fn appraise_verbose(&self, text: &str) -> AppraiseVerdict {
    let base = self.appraise(text);

    // Categorize evidence tokens dengan reason
    let mut support = Vec::new();
    let mut conflict = Vec::new();

    for (token, score) in &base.evidence {
        let reason = if let Some(&id) = self.token_to_id.get(token.as_str()) {
            let has_compositions = self.senses.get(&id)
                .map(|sm| sm.senses.iter().any(|s| !s.compositions.is_empty()))
                .unwrap_or(false);
            let is_seed = self.graph.get_node(id)
                .map(|n| n.is_seed)
                .unwrap_or(false);

            if is_seed { "seed".to_string() }
            else if has_compositions { "structural".to_string() }
            else { "cooccurrence".to_string() }
        } else {
            "novel".to_string()
        };

        if *score > 0.4 {
            support.push((token.clone(), *score, reason));
        } else {
            conflict.push((token.clone(), *score, reason));
        }
    }

    // Sort by score
    support.sort_by(|a, b| b.1.total_cmp(&a.1));
    conflict.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Generate human-readable explanation
    let explanation = {
        let n_support = support.len();
        let n_conflict = conflict.len();
        let support_tokens: Vec<String> = support.iter()
            .take(3)
            .map(|(t, s, r)| format!("{} ({}, {:.2})", t, r, s))
            .collect();
        let conflict_tokens: Vec<String> = conflict.iter()
            .take(3)
            .map(|(t, s, r)| format!("{} ({}, {:.2})", t, r, s))
            .collect();

        let mut parts = Vec::new();
        if n_support > 0 {
            parts.push(format!("{} token(s) support: {}",
                n_support, support_tokens.join(", ")));
        }
        if n_conflict > 0 {
            parts.push(format!("{} token(s) conflict: {}",
                n_conflict, conflict_tokens.join(", ")));
        }
        if parts.is_empty() {
            "No grounded tokens found — statement is novel to this graph.".to_string()
        } else {
            parts.join(". ") + "."
        }
    };

    let confidence_gap = base.agree_pct - base.disagree_pct;

    AppraiseVerdict {
        verdict: base.verdict,
        agree_pct: base.agree_pct,
        disagree_pct: base.disagree_pct,
        support,
        conflict,
        explanation,
        is_contextual: false,
        confidence_gap,
    }
}

/// Contextual verbose appraise — isolated, graph untouched.
pub fn appraise_against_verbose(&self, context: &str, statement: &str) -> AppraiseVerdict {
    let mut temp = Rsvs::new(self.config.clone()).expect("temp rsvs");
    let _ = temp.ingest_text(context);
    let mut verdict = temp.appraise_verbose(statement);
    verdict.is_contextual = true;
    verdict
}
```

---

## TASK 2 — `SessionGraph` struct — Dual Memory untuk RSVS

Terinspirasi dari Losion's `DualMemorySystem`:
- Losion: `WorkingMemory` (ring buffer, high detail) + `LongTermMemory` (compressed, persistent)
- RSVS: `SessionGraph` (volatile, per-session) + main `Rsvs` (persistent, long-term)

Buat file baru: `backend/crates/rsvs-core/src/session.rs`

```rust
//! SessionGraph — Working Memory layer untuk RSVS.
//!
//! Terinspirasi dari Losion's DualMemorySystem (Two-Level Memory):
//! - Losion WorkingMemory: ring buffer, high detail, volatile per-session
//! - Losion LongTermMemory: compressed AttnRes state, persistent
//!
//! Di RSVS:
//! - SessionGraph (ini): temporary Rsvs instance per context window
//!   Volatile — tidak persist ke disk. Dipakai untuk appraise_against().
//! - Main Rsvs: long-term graph, persistent, di-ingest dari corpus besar
//!
//! Pattern: main graph adalah "what we know long-term".
//! SessionGraph adalah "what this specific text tells us right now".
//! Verdict dari SessionGraph bisa berbeda dari main graph — itu feature, bukan bug.

use crate::{AppraiseVerdict, PipelineConfig, Rsvs, RsvsError};

/// A temporary, isolated knowledge session.
///
/// Analogous to Losion's WorkingMemory: stores recent context with
/// full fidelity, volatile (cleared when dropped), limited scope.
pub struct SessionGraph {
    inner: Rsvs,
    context_text: String,
    sentences_ingested: usize,
    atoms_induced: usize,
}

impl SessionGraph {
    /// Create a new session with the given context text.
    /// All knowledge is auto-induced from the context — no manual composition.
    pub fn new(context: &str, config: PipelineConfig) -> Result<Self, RsvsError> {
        let mut inner = Rsvs::new(config)?;
        let stats = inner.ingest_text(context)?;
        Ok(Self {
            inner,
            context_text: context.to_string(),
            sentences_ingested: stats.sentences_processed,
            atoms_induced: stats.atoms_promoted,
        })
    }

    /// Appraise a statement against this session's context only.
    pub fn appraise(&self, statement: &str) -> AppraiseVerdict {
        let mut verdict = self.inner.appraise_verbose(statement);
        verdict.is_contextual = true;
        verdict
    }

    /// Compare two statements — returns which is better supported by context.
    /// Useful for contradiction detection: statement_a (TRUE) vs statement_b (FALSE).
    pub fn compare(&self, statement_a: &str, statement_b: &str) -> SessionComparison {
        let va = self.appraise(statement_a);
        let vb = self.appraise(statement_b);

        let winner = if va.agree_pct > vb.agree_pct {
            ComparisonWinner::A
        } else if vb.agree_pct > va.agree_pct {
            ComparisonWinner::B
        } else {
            ComparisonWinner::Tied
        };

        let gap = (va.agree_pct - vb.agree_pct).abs();
        let is_discriminable = gap > 10.0; // >10pp gap = clearly discriminable

        SessionComparison {
            verdict_a: va,
            verdict_b: vb,
            winner,
            agree_gap: gap,
            is_discriminable,
            explanation: format!(
                "Gap: {:.1}pp. {}",
                gap,
                if is_discriminable {
                    "Clearly discriminable."
                } else {
                    "Ambiguous — more context needed."
                }
            ),
        }
    }

    /// Session stats — how much was induced from context
    pub fn stats(&self) -> SessionStats {
        let status = self.inner.status();
        SessionStats {
            sentences_ingested: self.sentences_ingested,
            atoms_induced: self.atoms_induced,
            total_nodes: status.total_nodes,
            total_atoms: status.total_atoms,
        }
    }

    /// Context text that was used to build this session
    pub fn context(&self) -> &str {
        &self.context_text
    }
}

#[derive(Debug)]
pub enum ComparisonWinner {
    A,
    B,
    Tied,
}

#[derive(Debug)]
pub struct SessionComparison {
    pub verdict_a: AppraiseVerdict,
    pub verdict_b: AppraiseVerdict,
    pub winner: ComparisonWinner,
    pub agree_gap: f32,
    pub is_discriminable: bool,
    pub explanation: String,
}

#[derive(Debug)]
pub struct SessionStats {
    pub sentences_ingested: usize,
    pub atoms_induced: usize,
    pub total_nodes: usize,
    pub total_atoms: usize,
}
```

Tambahkan ke `backend/crates/rsvs-core/src/lib.rs`:
```rust
pub mod session;
pub use session::{SessionGraph, SessionComparison, SessionStats, ComparisonWinner};
```

---

## TASK 3 — Update `rsvs-realtest.rs` pakai SessionGraph + appraise_verbose

File: `backend/crates/rsvs-core/src/bin/rsvs-realtest.rs`

Di PART 3 (Contextual Appraise), ganti `rsvs.appraise_against()` dengan `SessionGraph`:

```rust
use rsvs::session::SessionGraph;

// PART 3: Contextual Appraise via SessionGraph (Dual Memory pattern)
section("PART 3: SessionGraph — Dual Memory (Working Graph)");

let session = SessionGraph::new(context_andi, make_config())
    .expect("session failed");

println!("\n  Session stats: {} sentences, {} atoms induced",
    session.stats().sentences_ingested,
    session.stats().atoms_induced);

// Gunakan compare() untuk contradiction detection
let comparison = session.compare(
    "Andi mengajar di sekolah",
    "Andi menangkap ikan di laut",
);
println!("\n  Comparison: {}", comparison.explanation);
println!("  TRUE  ({:.1}% agree): {}", comparison.verdict_a.agree_pct,
    comparison.verdict_a.explanation);
println!("  FALSE ({:.1}% agree): {}", comparison.verdict_b.agree_pct,
    comparison.verdict_b.explanation);
println!("  Discriminable: {}", comparison.is_discriminable);

// Verify graph untouched
let nodes_before = rsvs.status().total_nodes;
// SessionGraph is dropped here — main graph untouched
drop(session); // explicit drop for clarity
let nodes_after = rsvs.status().total_nodes;
assert_eq!(nodes_before, nodes_after, "MAIN GRAPH MODIFIED — isolation broken!");
println!("\n  ISOLATION VERIFIED: main graph untouched ({} nodes)", nodes_after);
```

Di PART 2 (Auto-Induced Appraise), ganti `print_appraise()` dengan `appraise_verbose()`:
```rust
// Contoh untuk statement 1:
let v1 = rsvs.appraise_verbose("Budi bekerja di rumah sakit sebagai dokter");
println!("\n  Statement 1 (TRUE): '{}'", "Budi bekerja di rumah sakit sebagai dokter");
println!("  Verdict: {} ({:.1}% / {:.1}%) gap={:.1}pp",
    v1.verdict, v1.agree_pct, v1.disagree_pct, v1.confidence_gap);
println!("  Explanation: {}", v1.explanation);
```

---

## TASK 4 — Update TUI untuk tampilkan verbose verdict

File: `backend/crates/rsvs-core/src/bin/rsvs-tui.rs`

Di APPRAISE mode output, tampilkan `appraise_verbose()` bukan `appraise()`:

```
Verdict : agree (73.2% / 13.1%)  gap: 60.1pp  [CONFIDENT]
Support : budi (structural, 0.91) | dokter (structural, 0.87) | rumah (cooccurrence, 0.72)
Conflict: sawah (novel, 0.05) | padi (novel, 0.08)
────────────────────────────────────────────────
3 token(s) support: budi (structural, 0.91), dokter (structural, 0.87), rumah (cooccurrence, 0.72).
1 token(s) conflict: sawah (novel, 0.05).
```

Di CONTEXT mode, gunakan `SessionGraph`:
```
[Session] 10 sentences ingested → 22 atoms induced
[Context] Andi adalah seorang guru yang mengajar di sekolah...
────────────────────────────────────────────────
Statement: Andi mengajar di sekolah
Verdict  : agree (61.3% / 8.2%)  gap: 53.1pp  [CONTEXTUAL]
Explanation: 3 token(s) support: andi (structural, 0.88)...
```

Label `[CONTEXTUAL]` muncul kalau `is_contextual = true`.
Label `[CONFIDENT]` kalau `confidence_gap > 30.0`.
Label `[AMBIGUOUS]` kalau `confidence_gap < 10.0`.

---

## TASK 5 — Build, test, commit

```bash
cd backend
cargo build --release --bin rsvs-realtest --bin rsvs-tui 2>&1 | grep -E "^error" | head -20
./target/release/rsvs-realtest
```

Lalu commit:
```bash
git add \
  backend/crates/rsvs-core/src/session.rs \
  backend/crates/rsvs-core/src/pipeline/modes.rs \
  backend/crates/rsvs-core/src/lib.rs \
  backend/crates/rsvs-core/src/bin/rsvs-realtest.rs \
  backend/crates/rsvs-core/src/bin/rsvs-tui.rs

git commit -m "feat: Losion bridge — AppraiseVerdict + SessionGraph (Dual Memory)

Adapted from Losion's NeuroSymbolicVerifier + DualMemorySystem:
- AppraiseVerdict: token-level explanation with reason (structural/seed/cooccurrence/novel)
- appraise_verbose(): returns AppraiseVerdict with human-readable explanation
- appraise_against_verbose(): contextual verbose appraise, graph untouched
- SessionGraph: formal Dual Memory working layer, volatile per-context
- SessionGraph::compare(): contradiction detection with discriminability check
- TUI: shows confidence_gap + [CONFIDENT/AMBIGUOUS/CONTEXTUAL] labels

Source references in losion_bridge/ (to be deleted after implementation)."

git push origin main
```

---

## TASK 6 — Hapus folder losion_bridge setelah implementasi selesai

```bash
rm -rf losion_bridge/
git add -A
git commit -m "chore: remove losion_bridge reference files (implementation complete)"
git push origin main
```

---

## Summary: Apa yang Kita Ambil dari Losion

| Losion Concept | RSVS Adaptation |
|---|---|
| `VerificationStatus` (VERIFIED/FAILED/PARTIAL) | `AppraiseVerdict.verdict` + confidence_gap |
| `VerificationResult.error_type` | `AppraiseVerdict.conflict` tokens dengan reason |
| `VerificationResult.feedback` | `AppraiseVerdict.explanation` (human-readable) |
| `WorkingMemory` (ring buffer, volatile) | `SessionGraph` (temp Rsvs, volatile) |
| `LongTermMemory` (compressed, persistent) | Main `Rsvs` graph (persistent) |
| `DualMemorySystem.consolidate()` | Future: `SessionGraph → main graph` promotion |
| `ReflectionEngine` (verbal feedback) | `AppraiseVerdict.explanation` (per-verdict) |
| `Reflection.lesson` | Future: store contradiction patterns ke main graph |

Yang TIDAK diambil (tidak relevan untuk symbolic system):
- Neural network weights (nn.Linear, nn.Parameter)
- PyTorch tensors dan gradient flow
- Tri-Jalur SSM/Attention/MoE routing
- Differentiable verification scores
