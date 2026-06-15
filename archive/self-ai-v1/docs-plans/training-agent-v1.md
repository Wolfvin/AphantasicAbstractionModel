# Plan: TrainingAgent v1 — Explicit Intent CLI

**Identity:** Planner·Thorough
**Alasan Dibuat:** Koreksi otomatis via regex (`_detect_correction()`) terbukti fragile. User meminta pendekatan explicit intent — user harus sengaja memicu training, bukan sistem menebak dari teks. TrainingAgent adalah dedicated CLI module untuk mengajari SELF dari interaksi nyata, bukan simulasi.

---

## Ringkasan

Bangun `self-ai/src/training/` — satu modul dedicated yang tugasnya mengajari SELF dari interaksi nyata. Entry point: `python -m self_ai.training`. Deteksi otomatis dihapus, diganti dengan intent eksplisit.

---

## 1. Yang Dihapus (engine.py)

| Item | Lokasi | Alasan |
|------|--------|--------|
| `_CORRECTION_PATTERNS` | engine.py L307-338 | 6 regex pattern — fragile, false positive prone |
| `_detect_correction()` | engine.py L346-374 | Auto-detect dari teks — terlalu fragile |
| `_handle_correction()` | engine.py L383-417 | Routing layer untuk auto-detect — tidak diperlukan lagi |
| Pre-check di `derive_from_text()` | engine.py L186-192 | `correction = self._detect_correction(question)` + routing — hapus |
| `@FLOW: CORRECTION_DETECT` | AGENTS.md L786 | Flow sudah tidak ada |

**File header engine.py** perlu diupdate — hapus referensi correction detection.

**Komentar `derive_from_text()`** perlu diupdate — hapus mention v42 correction pre-check.

---

## 2. Yang Dipertahankan (text_comprehension.py)

| Item | Lokasi | Alasan |
|------|--------|--------|
| `teach_from_correction()` | text_comprehension.py L1857-1926 | Core logic — dipanggil dari TrainingAgent |
| `_generate_correction_reasoning()` | text_comprehension.py L1936-1977 | Qwen3 reasoning generation — dipanggil oleh teach_from_correction() |
| `_save_learned_patterns()` | text_comprehension.py L2612-2635 | Persistence — sudah ada |
| `_load_learned_patterns()` | text_comprehension.py L2280-2334 | Load dari disk — sudah ada |
| `provide_feedback()` | text_comprehension.py L2336+ | External validation — tetap dipakai |

---

## 3. Yang Dibangun Baru

### Struktur File

```
self-ai/src/training/
├── __init__.py          ← kosong, package marker
├── __main__.py          ← entry point: python -m self_ai.training
├── training_agent.py    ← core: run(), correct(), benchmark(), export()
├── session.py           ← track satu sesi training
└── results.py           ← export ke docs/training_sessions/
```

### 3.1 `session.py` — Session Tracking

```python
class TrainingSession:
    """Track satu sesi training — semua soal, jawaban, koreksi, reasoning."""
    
    def __init__(self):
        self.started_at = datetime.now()
        self.questions: list[QuestionResult] = []
        self.corrections: list[CorrectionRecord] = []
        self.benchmark_before: dict | None = None
        self.benchmark_after: dict | None = None
    
    def add_question(self, context, question, answer, confidence, method):
        """Catat hasil satu pertanyaan."""
    
    def add_correction(self, question, wrong_answer, correct_answer, reasoning, pattern_key):
        """Catat satu koreksi yang diterapkan."""
    
    def set_benchmark(self, phase, results):
        """Set benchmark results: phase='before' atau 'after'."""
    
    def summary(self) -> dict:
        """Return summary dict untuk export."""
```

### 3.2 `training_agent.py` — Core Logic

```python
class TrainingAgent:
    """Agent yang mengajari SELF dari interaksi nyata."""
    
    def __init__(self):
        self.engine = DerivationEngine(SelfCore())
        self.engine._init_modules()
        self.tc = self.engine.text_comprehension
        self.session = TrainingSession()
        self._last_result = None  # Untuk correct() yang merujuk run() terakhir
    
    def run(self, question: str, context: str) -> dict:
        """Jalankan soal ke SELF, catat hasilnya.
        
        Returns: {answer, confidence, method, pattern_used}
        """
        result = self.engine.derive_from_text(context, question)
        self._last_result = {
            'context': context,
            'question': question,
            'answer': result.get('answer'),
            'confidence': result.get('confidence', 0),
            'method': result.get('method', ''),
        }
        self.session.add_question(context, question, 
            result.get('answer'), result.get('confidence', 0), result.get('method', ''))
        return self._last_result
    
    def correct(self, correct_answer: str) -> dict:
        """Terima koreksi untuk run() terakhir.
        
        1. Ambil question+context dari run() terakhir
        2. Generate reasoning via Qwen3
        3. Return reasoning untuk konfirmasi user
        4. Jika dikonfirmasi → teach_from_correction()
        
        Returns: {reasoning, pattern_key, confirmed}
        """
        if self._last_result is None:
            return {'error': 'No question to correct. Run (q)uestion first.'}
        
        # Generate reasoning via Qwen3
        reasoning = self.tc._generate_correction_reasoning(
            question=self._last_result['question'],
            correct_answer=correct_answer,
            context_text=self._last_result['context'],
        )
        
        return {
            'reasoning': reasoning,
            'correct_answer': correct_answer,
            'question': self._last_result['question'],
            'confirmed': False,  # Menunggu konfirmasi user
        }
    
    def confirm_correction(self, correct_answer: str, reasoning: str = '') -> dict:
        """Konfirmasi koreksi → panggil teach_from_correction()."""
        if self._last_result is None:
            return {'error': 'No question to correct.'}
        
        result = self.tc.teach_from_correction(
            text=self._last_result['context'],
            question=self._last_result['question'],
            correct_answer=correct_answer,
            correction_raw=f"manual correction: {correct_answer}",
        )
        
        self.session.add_correction(
            self._last_result['question'],
            self._last_result['answer'],
            correct_answer,
            reasoning or result.get('reasoning', ''),
            result.get('pattern_key', ''),
        )
        
        return {
            'pattern_key': result.get('pattern_key', ''),
            'reasoning': reasoning or result.get('reasoning', ''),
            'confirmed': True,
        }
    
    def benchmark(self, test_cases: list = None) -> dict:
        """Ukur accuracy before/after."""
        # Reuse logic dari benchmark_empiris.py
        ...
    
    def export_session(self) -> str:
        """Export ke docs/training_sessions/YYYY-MM-DD_HH-MM.md"""
        ...
```

### 3.3 `results.py` — Auto-Documentation

```python
def export_session(session: TrainingSession, output_dir: str) -> str:
    """Export session ke Markdown file.
    
    Output format:
    # Training Session [timestamp]
    
    ## Summary
    - Total corrections: N
    - Accuracy before: X%
    - Accuracy after: Y%
    - Delta: +Z%
    
    ## Corrections Made
    ### Correction 1
    - Question: ...
    - Context: ...
    - SELF answered: ...
    - Correct answer: ...
    - Reasoning saved: ...
    - Pattern key: ...
    
    ## Benchmark Results
    [before/after per domain]
    """
```

### 3.4 `__main__.py` — CLI Interface

```
=== SELF Training Agent ===
(q)uestion  (c)orrect  (b)enchmark  (e)xport  (x)exit

> q
Konteks: [user input]
Pertanyaan: [user input]
SELF: [jawaban] (confidence: X.XX)

> c
Jawaban benar: [user input]
Reasoning: "[generated oleh Qwen]"
Tepat? (y/edit/n):
→ teach_from_correction() dipanggil hanya setelah konfirmasi eksplisit

> b
Running benchmark...

> e
→ export ke docs/training_sessions/YYYY-MM-DD.md otomatis

> x
Exit (auto-export jika ada data)
```

---

## 4. Benchmark Integration

TrainingAgent.benchmark() harus reusable. Strategi:

1. **Extract** test cases dari `benchmark_empiris.py` (TEST_SOAL) ke shared module
2. TrainingAgent.benchmark() pakai test cases yang sama
3. Return `{before: {total, per_type}, after: {total, per_type}, delta}`

Opsi implementasi:
- **Minimal**: Import langsung dari `benchmark_empiris.py` — sudah ada `TEST_SOAL` dan `check_answer()`
- **Better**: Pindahkan `TEST_SOAL` dan `check_answer()` ke `benchmark/test_data.py` — shared, tapi perlu refactor

**Keputusan:** Minimal dulu — import dari benchmark_empiris. Jangan over-engineer.

---

## 5. Proof of Concept yang Harus Dibuktikan

Satu session nyata harus membuktikan tiga hal:

1. **SELF jawab salah → user koreksi → reasoning di-generate → dikonfirmasi → tersimpan**
   - Flow: q → SELF jawab salah → c → user kasih jawaban benar → reasoning muncul → y → pattern tersimpan

2. **Pattern survive restart**
   - Setelah teach_from_correction(), data ada di `data/learned_patterns.json`
   - Bisa diverifikasi dengan load ulang

3. **Accuracy setelah koreksi lebih tinggi dari sebelumnya**
   - benchmark() sebelum dan sesudah koreksi
   - Delta harus positif (setidaknya untuk test case yang dikoreksi)

---

## 6. Dependency Flow

```
__main__.py
  → TrainingAgent (training_agent.py)
    → DerivationEngine (engine.py) — derive_from_text()
    → TextComprehension (text_comprehension.py)
      → teach_from_correction()
      → _generate_correction_reasoning()
      → _save_learned_patterns()
    → TrainingSession (session.py)
    → export_session (results.py)
```

Tidak ada dependency baru selain yang sudah ada. TrainingAgent hanya memanggil API yang sudah ada.

---

## 7. Urutan Implementasi

1. **Update AGENTS.md** — catat v2 note tentang chat session layer (sudah ada di v2 backlog)
2. **Hapus correction detection** dari engine.py
3. **Bangun session.py** — data class untuk tracking
4. **Bangun training_agent.py** — core logic
5. **Bangun results.py** — export ke markdown
6. **Bangun __main__.py** — CLI interface
7. **Test manual** — jalankan session nyata
8. **Commit & push**

---

## 8. Risk & Mitigasi

| Risk | Mitigasi |
|------|----------|
| teach_from_correction() dipanggil tanpa konfirmasi | CLI flow: correct() → tampilkan reasoning → konfirmasi → confirm_correction() |
| Benchmark memakan waktu (model loading) | Benchmark opsional, user trigger manual via (b) |
| Pattern conflict — koreksi menimpa pattern yang sudah ada | teach() sudah handle via content hash di pattern key |
| Qwen3 reasoning terlalu lambat | Fallback ke generic reasoning — sudah ada di _generate_correction_reasoning() |

---

## 9. Yang TIDAK Termasuk Scope v1

- Web/UI interface — hanya CLI
- Auto-detection koreksi dari chat — dihapus, diganti explicit intent
- Chat session layer — dicatat di AGENTS.md untuk v2
- Batch training dari file — v2
- Undo koreksi — v2
