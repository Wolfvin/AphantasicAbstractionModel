# AAM Roadmap

## v1.0.0 (Current) — Rule-Based Release

All knowledge is hardcoded: 13 transforms, templates, lifecycle rules, and epistemic
logic are written by us in Rust. AAM can **execute** knowledge but cannot **discover**
knowledge on its own. It is a perfect calculator — but one that never asks "why is this
formula like this?"

- 6 Unified Abstractions (SemanticAtom, Composition, LifecycleState+EpistemicState,
  SemanticEdge, Transform DAG, Seed Anchoring)
- 13-Transform DAG Pipeline with condition-gated execution
- Compositional Verbalization Engine (CVE) — zero-hallucination graph-driven explanation
- Spreading Activation, Convergence Detection, Temporal Decay
- Persistence (JSON save/load)
- PyO3 Python bindings with full pipeline access
- 142 tests passing

---

## v2.0.0 (Planned) — Reflexive Layer: Self-Learning Bootstrap

The key gap: AAM cannot observe its own processing. It "sees" the world, but it does
not "see" itself seeing the world. The Reflexive Layer closes this gap with one new
concept: **ReflexiveComposition** — a composition whose subject is another composition
or a transform.

### New Components

| Component | Purpose | Est. Lines |
|-----------|---------|------------|
| `ReflexiveComposition` type | Composition about compositions/transforms | ~50 |
| `AskWhy` transform | Sits at end of pipeline, detects patterns in meta-layer | ~150 |
| Meta-layer logging | Every transform auto-logs MetaComposition | ~80 |
| Soft transform registry | Stable Hypotheses become virtual transforms (JSON-driven) | ~120 |
| `reflexive.rs` module | New module in `v12/` | ~400 total |

### Architecture

```
Layer 0 (existing):  Input → EnrichComposition → HiddenMeaning → Pattern → ...
                       "AAM memproses dunia"

Layer 1 (new):       Setiap transform yang jalan → buat MetaComposition
                       { source_composition, transform_id, result_composition, energy_delta }
                       "AAM mengamati dirinya memproses dunia"

Layer 2 (emergent):  ConvergenceDetection di Layer 1 → "Hei, transform X
                       selalu dipicu oleh pola Y" → Hypothesis composition
                       "AAM belajar dari pengamatannya terhadap dirinya sendiri"
```

### Learning Cycle

```
Input masuk
    ↓
Pipeline 13+ transforms (hardcoded + soft)
    ↓
Setiap transform → log MetaComposition (Layer 1)
    ↓
AskWhy → cek pola di meta-layer
    ↓
Pola ditemukan? → Hypothesis (lifecycle=Candidate, confidence=low)
    ↓
Input berikutnya masuk
    ↓
Hypothesis diuji: apakah prediksinya cocok?
    ↓
Cocok? → confidence++ → Candidate→Inferred→Grounded→Stable
Tidak? → confidence-- → mungkin dihapus
    ↓
Stable Hypothesis → jadi "soft transform" baru
    ↓
AAM sekarang punya aturan yang DIA SENDIRI temukan
    ↓
═══ SELF-HOSTING BOOTSTRAP ═══
```

### Key Properties

- **No LLM required** — Uses existing SpreadingActivation, ConvergenceDetection, and
  Lifecycle mechanisms
- **Zero hallucination preserved** — Hypotheses are grounded in observed meta-patterns
- **Fully auditable** — Every self-discovered rule has a traceable provenance chain
- **No GPU needed** — Pure graph traversal + pattern matching
- **Backward compatible** — All v1.0.0 transforms remain hardcoded; new rules are additive

### Prerequisites (already in v1.0.0)

| Mechanism | Status | Role in v2.0.0 |
|-----------|--------|-----------------|
| Lifecycle promotion | ✅ Mature | Candidate→Stable for Hypotheses |
| SpreadingActivation | ✅ Mature | Energy flow for meta-patterns |
| ConvergenceDetection | ✅ Ready | Pattern detection in meta-layer |
| Pipeline DAG | ✅ Mature | AskWhy as new transform node |
| Persistence | ✅ Ready | Save/learned soft transforms |

### What Needs To Be Built

1. **ReflexiveComposition type** — composition that refers to another composition/transform
2. **Meta-layer logging** — auto-generate MetaComposition per transform execution
3. **AskWhy transform** — one transform at end of pipeline
4. **Soft transform registry** — JSON-driven virtual transforms from Stable Hypotheses
5. **Data volume** — pipeline must receive enough input for patterns to emerge

---

## v3.0.0 (Future) — Self-Hosting

AAM discovers >50% of new rules on its own. Hardcoded transforms become the minority.
The "Rust built by Rust" moment — the system writes its own rules using the mechanisms
it originally inherited from us.

### Milestones

- AAM generates its first independently-discovered transform rule
- Self-discovered rules outperform original hardcoded rules in accuracy
- Full audit trail: every rule has provenance from observation → hypothesis → validation
- Tax compliance domain: AAM learns tax rules from regulation text without manual coding

### Open Questions

- What is the minimum data volume for reliable pattern emergence?
- How to prevent pathological self-reinforcing loops?
- Should soft transforms be promoted to Rust code (compiled) or stay JSON-driven?
- What is the validation protocol for self-discovered rules in production?

---

## Analogi: Bootstrap

```
Rust Bootstrapping                    AAM Bootstrapping
─────────────────                     ─────────────────
Stage 0: rustc (OCaml)          →     v1.0.0: 13 hardcoded transforms
  "compiler dari luar"                 "rules dari kita"
                                       
Stage 1: rustc (Rust)           →     v2.0.0: AskWhy + ReflexiveLayer
  "compiler menulis compiler"           "AAM menemukan pola di dirinya"
                                       
Stage 2: rustc compiles itself   →     v3.0.0: Self-Hosting
  "self-hosting"                        "AAM menulis aturannya sendiri"
```

Kita saat ini di Stage 0. Infrastruktur untuk Stage 1 sudah ada — kita cuma perlu
sambungkan satu lapis: **meta-composition**.
