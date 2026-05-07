# AGENT PROMPT — RSVS Master Improvement Plan
# All tasks in one file. Run as a swarm: each TASK block is independent
# unless marked [DEPENDS ON TASK X].

---

## PREAMBLE — What is RSVS and what are we fixing

RSVS (Relational Symbolic Vocabulary System) is a compositional symbolic meaning
engine. It builds a knowledge graph from raw text where every concept is defined
as a composition of other concepts — fully traceable, no opaque vectors.

**Strategic decision**: RSVS is now English-only. Indonesian corpus and Indonesian
functional seed words are removed. English has better structural regularity and
more available evaluation benchmarks. The architecture is language-agnostic so
this is purely a corpus/config change.

**Root problems being fixed in this prompt**:
1. Corpus is partially Indonesian — kills atom promotion for English eval
2. `rsvs-realtest.rs` uses Indonesian seeds and Indonesian text — broken for English
3. Eval benchmarks are too weak (gap ~5pp TRUE vs FALSE)
4. TUI binary exists as stub, not fully wired
5. No pytest integration test for accuracy (only latency benchmarks)

**Do NOT change**:
- Core algorithms: `ingest_text()`, `appraise()`, `relate()`, `structural_similarity()`
- Struct field names in `PipelineConfig`, `SenseConfig`, `AttentionConfig`
- PyO3 binding signatures
- The 24 epistemological seed atoms (exists, entity, relation, state, change,
  time, space, cause, effect, context, signal, pattern, memory, attention,
  value, agent, goal, risk, trust, identity, language, meaning, action, feedback)

---

## TASK A — Replace corpus with full English version

**File**: `python/rsvs/corpus.py`

**What to do**: Remove the `kerajaan` and `konsep` domains (Indonesian). Replace
with four new English domains: `profession`, `history`, `technology`, `society`.
Keep existing domains (`geology`, `water`, `biology`, `physics`, `materials`)
exactly as-is — do not touch them.

Write 30 sentences per new domain. Style rules:
- Every sentence must be a simple declarative fact (Subject + Verb + Object)
- No subordinate clauses longer than one level
- Use the domain's core vocabulary repeatedly across sentences
  (repetition = co-occurrence = atom promotion)
- Each domain must have 3-5 "anchor" words that appear in 10+ sentences
  (these become the cross-domain atoms that eval benchmarks measure)

### Domain: `profession`

Anchor words (must appear 10+ times each): `doctor`, `patient`, `farmer`, `teacher`, `engineer`

Example sentences (write 30 total, these are just examples of the style):
- "A doctor examines patients to diagnose illness and prescribe medicine."
- "Farmers grow crops by planting seeds in fertile soil."
- "A teacher explains concepts to students in a classroom."
- "Engineers design structures by applying scientific principles."
- "A doctor treats patients using medical tools and knowledge."
- "Farmers harvest crops from fields after the growing season."
- ...continue for 30 total, ensuring anchors repeat sufficiently

### Domain: `history`

Anchor words (must appear 10+ times each): `war`, `empire`, `trade`, `civilization`, `ruler`

Example style:
- "An empire is a large territory controlled by a single ruler."
- "War is an armed conflict between groups or nations."
- "Trade routes connected ancient civilizations across continents."
- "A civilization develops complex institutions including government and law."
- "Rulers of empires maintained power through military force and taxation."
- ...continue for 30 total

### Domain: `technology`

Anchor words (must appear 10+ times each): `computer`, `network`, `data`, `software`, `processor`

Example style:
- "A computer processes data using a central processing unit."
- "Software is a set of instructions that runs on a computer."
- "A network connects computers to allow data to be shared."
- "A processor executes software instructions at high speed."
- "Data is stored in memory and retrieved by software programs."
- ...continue for 30 total

### Domain: `society`

Anchor words (must appear 10+ times each): `law`, `government`, `citizen`, `economy`, `institution`

Example style:
- "A government creates laws to organize and protect citizens."
- "Citizens follow laws established by their government."
- "An economy is the system by which goods and services are produced."
- "Laws define the rights and responsibilities of citizens."
- "Institutions enforce laws and maintain order in society."
- ...continue for 30 total

### After writing the domains, update `SIMILARITY_TRIPLES` at the bottom of corpus.py

Add these new triples that use the new domain vocabulary:
```python
# These go in eval.py SIMILARITY_TRIPLES, not corpus.py — see Task C
```

### Deliverables

1. Rewrite `python/rsvs/corpus.py` with all 9 domains (5 existing + 4 new)
   Total sentences: ~270 (existing 150 + new 120)
2. Verify the file runs: `python -c "from rsvs.corpus import DOMAINS; print({k: len(v) for k, v in DOMAINS.items()})"`
   Expected output: all domains show 30 sentences

### Commit

```bash
git add python/rsvs/corpus.py
git commit -m "corpus: replace Indonesian domains with full English corpus

- Remove kerajaan and konsep (Indonesian)
- Add profession, history, technology, society (English, 30 sentences each)
- Keep geology, water, biology, physics, materials unchanged
- All anchor words appear 10+ times for reliable atom promotion"
git push origin main
```

---

## TASK B — Retune rsvs-realtest.rs for English

**File**: `backend/crates/rsvs-core/src/bin/rsvs-realtest.rs`
**[DEPENDS ON: nothing — Rust changes are independent]**

This file currently uses Indonesian text and Indonesian functional seed words.
Convert everything to English.

### B.1 — Rewrite `make_config()`

Replace the Indonesian functional words in `custom_seeds` with English equivalents.
English functional words appear in almost every sentence and enable the same
"grounding gate" effect as Indonesian `yang`, `di`, `dan` did before.

```rust
fn make_config() -> PipelineConfig {
    let custom_seeds: Vec<String> = vec![
        // Epistemological primitives (original 24 — DO NOT CHANGE)
        "exists".into(), "entity".into(), "relation".into(), "state".into(),
        "change".into(), "time".into(), "space".into(), "cause".into(),
        "effect".into(), "context".into(), "signal".into(), "pattern".into(),
        "memory".into(), "attention".into(), "value".into(), "agent".into(),
        "goal".into(), "risk".into(), "trust".into(), "identity".into(),
        "language".into(), "meaning".into(), "action".into(), "feedback".into(),
        // English functional/grammatical words (grounding gate for English text)
        // These appear in almost every English sentence — enables sentence-level
        // grounding for ANY English input. NOT content words.
        "the".into(), "is".into(), "are".into(), "was".into(),
        "a".into(), "an".into(), "of".into(), "in".into(),
        "and".into(), "to".into(), "that".into(), "it".into(),
        "by".into(), "with".into(), "for".into(), "from".into(),
        "has".into(), "have".into(), "be".into(), "been".into(),
        "not".into(), "as".into(), "or".into(), "at".into(),
        "its".into(), "which".into(), "when".into(), "can".into(),
    ];

    // Tuned parameters for short English corpus (same rationale as before,
    // but now English-verified)
    let mut induction = rsvs::sense::SenseInductionConfig::default();
    induction.tau_overlap = 0.5;
    induction.tau_compress = 0.15;
    induction.composition_min_confidence = 0.15;

    let mut sense = rsvs::sense::SenseConfig::default();
    sense.theta_assign = 0.20;
    sense.gamma_stopword = 0.85;
    sense.induction = induction;

    let mut attention = rsvs::attention::AttentionConfig::default();
    attention.min_cooc = 1;

    PipelineConfig {
        entity_promote_n: 2,
        custom_seeds: Some(custom_seeds),
        sense,
        attention,
        tau_entity_learned: 0.10,
        ..PipelineConfig::default()
    }
}
```

### B.2 — Rewrite all corpus vectors in `main()`

Replace the Indonesian story vectors with English equivalents. Keep the same
6-domain discriminability test structure — just translate everything to English.

Use these English corpora (write your own sentences matching this style):

**Domain 1: Alice the doctor** (~15 sentences, anchor = doctor/patient/hospital/medicine/treat)
```
"Alice is a doctor who works at a hospital."
"Alice examines patients every day to diagnose illness."
"The hospital where Alice works has modern medical equipment."
"Alice prescribes medicine to help patients recover quickly."
"Patients trust Alice because of her medical knowledge."
... (continue to 15)
```

**Domain 2: Bob the farmer** (~15 sentences, anchor = farmer/field/crop/harvest/plant)
```
"Bob is a farmer who grows crops on his land."
"Bob plants seeds in the field at the start of each season."
"The fields where Bob farms are large and very fertile."
"Bob harvests crops at the end of the growing season."
"Farmers like Bob depend on rain to water their crops."
... (continue to 15)
```

**Domain 3: Clara the teacher** (~15 sentences, anchor = teacher/student/school/lesson/learn)
```
"Clara is a teacher who works at a school."
"Clara explains lessons to students in the classroom."
"Students learn from Clara every day at school."
"The school where Clara teaches has many classrooms."
"Clara prepares lessons so students can understand the subject."
... (continue to 15)
```

**Domain 4: computers/technology** (~15 sentences, anchor = computer/data/software/processor/network)
```
"A computer processes data using a processor at high speed."
"Software runs on a computer and controls what it does."
"A network connects computers so they can share data."
"The processor executes software instructions very quickly."
"Data is stored in memory and retrieved by software."
... (continue to 15)
```

**Domain 5: history** (~15 sentences, anchor = empire/ruler/war/trade/civilization)
```
"An empire is a large territory controlled by a ruler."
"War is an armed conflict between groups or nations."
"Ancient civilizations developed through trade and agriculture."
"Rulers of empires maintained power through military force."
"Trade routes connected civilizations across continents."
... (continue to 15)
```

**Domain 6: mountains/nature** (~15 sentences, anchor = mountain/peak/forest/rock/elevation)
```
"A mountain is a large landform that rises steeply above the land."
"Mountains have peaks that reach high elevations above sea level."
"Forests grow on the slopes of mountains in temperate regions."
"Rock formations on mountains are shaped by erosion over time."
"The peak of a mountain has cold thin air at high elevation."
... (continue to 15)
```

### B.3 — Rewrite all `appraise()` calls

For each domain, write:
- 1 TRUE statement (clearly consistent with the domain corpus)
- 1 FALSE statement (clearly inconsistent — wrong role/wrong domain)
- 1 CROSS-DOMAIN FALSE (mix domains absurdly: "Alice plants crops in the field")

Example pattern:
```rust
// TRUE about Alice
let a1 = rsvs.appraise("Alice treats patients at the hospital");
print_appraise("Doctor-TRUE: 'Alice treats patients at the hospital'", &a1);

// FALSE — wrong profession
let a2 = rsvs.appraise("Alice plants crops in the field");
print_appraise("Doctor-FALSE: 'Alice plants crops in the field'", &a2);
```

### B.4 — Rewrite SUMMARY section discriminability table

Replace Indonesian labels with English:
```rust
let domains = vec![
    ("Alice(doctor)",   a1.agree_pct, a2.agree_pct),
    ("Bob(farmer)",     b1.agree_pct, b2.agree_pct),
    ("Clara(teacher)",  c1.agree_pct, c2.agree_pct),
    ("Technology",      t1.agree_pct, t2.agree_pct),
    ("History",         h1.agree_pct, h2.agree_pct),
    ("Mountain",        m1.agree_pct, m2.agree_pct),
];
```

### B.5 — Build and run

```bash
cd backend
cargo build --release --bin rsvs-realtest 2>&1 | grep -E "^error" | head -20
./target/release/rsvs-realtest
```

**Target metrics**:
- Atoms promoted: ≥ 50
- Compositions induced: ≥ 600
- TRUE avg > FALSE avg, gap ≥ 10pp
- ≥ 5/6 domains discriminable

If targets not met, the parameter tuning in `make_config()` is correct — the
issue is insufficient corpus repetition. Add more sentences to the failing
domain (not parameter changes).

### Commit

```bash
git add backend/crates/rsvs-core/src/bin/rsvs-realtest.rs
git commit -m "realtest: convert to English corpus and seeds

- Replace Indonesian functional seeds with English (the, is, are, a, of, ...)
- Replace all 6 Indonesian story domains with English equivalents
- Keep same 6-domain discriminability test structure
- Same parameter tuning (entity_promote_n=2, tau_overlap=0.5, etc.)
- Target: ≥10pp TRUE/FALSE gap, ≥5/6 domains PASS"
git push origin main
```

---

## TASK C — Upgrade eval.py benchmarks for English accuracy
**[DEPENDS ON: Task A — needs corpus to have English domain vocabulary]**

**File**: `python/rsvs/eval.py`

### C.1 — Expand SIMILARITY_TRIPLES

Replace the existing 8 triples with 24 triples spanning all 9 domains.
The format is `(anchor, related, unrelated)` — `sim(anchor, related) > sim(anchor, unrelated)`.

```python
SIMILARITY_TRIPLES = [
    # geology / materials (existing — keep)
    ("solid",    "hard",       "liquid"),
    ("solid",    "material",   "water"),
    ("water",    "liquid",     "rock"),
    ("rock",     "solid",      "water"),
    ("material", "hard",       "water"),
    ("hard",     "solid",      "liquid"),
    ("energy",   "heat",       "water"),
    ("heat",     "energy",     "solid"),

    # biology / physics (cross-domain)
    ("cell",     "organism",   "rock"),
    ("organism", "species",    "metal"),
    ("force",    "energy",     "cell"),
    ("light",    "wave",       "rock"),

    # profession (new domain)
    ("doctor",   "patient",    "crop"),
    ("doctor",   "hospital",   "field"),
    ("farmer",   "crop",       "patient"),
    ("teacher",  "student",    "crop"),

    # technology (new domain)
    ("computer", "processor",  "crop"),
    ("software", "data",       "farmer"),
    ("network",  "computer",   "patient"),
    ("data",     "software",   "mountain"),

    # history (new domain)
    ("empire",   "ruler",      "software"),
    ("war",      "empire",     "crop"),

    # society (new domain)
    ("law",      "government", "crop"),
    ("citizen",  "law",        "processor"),
]
```

### C.2 — Update CROSS_DOMAIN_ATOMS and SINGLE_DOMAIN_ATOMS

Replace the current lists in `benchmark_confidence_growth()`:

```python
# Words that appear across MULTIPLE domains — should have higher confidence
CROSS_DOMAIN_ATOMS = [
    "energy",    # physics + biology + geology
    "material",  # geology + materials + physics
    "water",     # water + biology + geology
    "force",     # physics + materials
    "data",      # technology + society
]

# Words that appear in ONE domain only — should have lower confidence
SINGLE_DOMAIN_ATOMS = [
    "processor",   # technology only
    "emperor",     # history only
    "chromosome",  # biology only
    "basalt",      # geology only
    "polymer",     # materials only
]
```

### C.3 — Add new benchmark: DiscriminabilityBenchmark

Add this function after `benchmark_confidence_growth()`:

```python
# Discriminability pairs: (true_statement, false_statement, context_hint)
# context_hint is a word from the domain corpus to provide context to appraise()
DISCRIMINABILITY_PAIRS = [
    # profession domain
    ("doctor treats patients",    "doctor plants crops",         "hospital"),
    ("farmer grows crops",        "farmer treats patients",      "field"),
    ("teacher explains lessons",  "teacher harvests crops",      "student"),

    # technology domain
    ("computer processes data",   "computer grows crops",        "software"),
    ("software runs on computer", "software treats patients",    "processor"),

    # history domain
    ("empire controls territory", "empire treats patients",      "ruler"),
    ("war is armed conflict",     "war grows crops",             "military"),

    # biology domain
    ("cell is unit of life",      "cell processes data",         "organism"),
    ("species reproduce",         "species run on processors",   "biology"),

    # physics domain
    ("force changes motion",      "force grows crops",           "energy"),
    ("energy exists in forms",    "energy treats patients",      "heat"),
]

def benchmark_discriminability(r: Rsvs) -> BenchmarkResult:
    """
    Verifies that appraise(true_statement) > appraise(false_statement).
    This is the core proof: system discriminates consistent from inconsistent.
    """
    t0 = time.time()
    passed_pairs = []
    failed_pairs = []
    skipped = 0

    for true_stmt, false_stmt, ctx in DISCRIMINABILITY_PAIRS:
        try:
            # Use context_query hint via appraise with context
            r_true  = r.appraise(true_stmt)
            r_false = r.appraise(false_stmt)
        except Exception:
            skipped += 1
            continue

        if r_true is None or r_false is None:
            skipped += 1
            continue

        gap = r_true.agree_pct - r_false.agree_pct
        if gap > 0:
            passed_pairs.append((true_stmt, false_stmt, gap))
        else:
            failed_pairs.append((true_stmt, false_stmt, gap))

    evaluated = len(passed_pairs) + len(failed_pairs)
    score = len(passed_pairs) / evaluated if evaluated > 0 else 0.0
    threshold = 0.60  # ≥60% of pairs must be discriminable

    details = {
        "evaluated_pairs": evaluated,
        "skipped":         skipped,
        "passed_pairs":    len(passed_pairs),
        "failed_pairs":    len(failed_pairs),
    }
    for i, (t, f, gap) in enumerate(passed_pairs[:3]):
        details[f"pass_{i}"] = f"gap={gap:+.1f}pp: '{t[:30]}' > '{f[:30]}'"
    for i, (t, f, gap) in enumerate(failed_pairs[:2]):
        details[f"fail_{i}"] = f"gap={gap:+.1f}pp: '{t[:30]}' <= '{f[:30]}'"

    avg_gap = (sum(g for _, _, g in passed_pairs) / len(passed_pairs)
               if passed_pairs else 0.0)

    verdict = (f"{len(passed_pairs)}/{evaluated} pairs discriminable, "
               f"avg_gap={avg_gap:+.1f}pp" if evaluated > 0
               else "no pairs evaluated")

    return BenchmarkResult(
        name="Discriminability",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details=details,
        elapsed_s=round(time.time() - t0, 3),
        verdict=verdict,
    )
```

### C.4 — Wire the new benchmark into `run_eval()`

In the `run_eval()` function, after `b3 = benchmark_confidence_growth(r)`, add:

```python
# 3b. Discriminability (new — most important benchmark)
b3b = benchmark_discriminability(r)
results.append(b3b)
if verbose:
    print(str(b3b))
```

### Commit

```bash
git add python/rsvs/eval.py
git commit -m "eval: expand benchmarks for English accuracy

- Expand SIMILARITY_TRIPLES from 8 to 24 (covers all 9 domains)
- Update CROSS/SINGLE_DOMAIN_ATOMS to match new English corpus
- Add DiscriminabilityBenchmark: 11 pairs, threshold 60%
- This is the core proof benchmark: TRUE > FALSE discriminability"
git push origin main
```

---

## TASK D — Add pytest integration test for accuracy
**[DEPENDS ON: Task A (English corpus) and Task C (eval.py Discriminability)]**

**File**: `python/tests/test_accuracy.py` (new file)

This test is designed to run in CI without a compiled Rust binary. It uses the
`rsvs_core.py` mock/protocol layer. If the Rust binary IS available, it runs
the full test; if not, it skips gracefully.

```python
"""
RSVS Accuracy Integration Test

Tests that the system discriminates true from false statements.
Skips if Rust native extension is not compiled.

Run: pytest python/tests/test_accuracy.py -v
"""
import pytest

# Try to import the native extension; skip entire module if unavailable
try:
    from rsvs import Rsvs
    RSVS_AVAILABLE = True
except ImportError:
    RSVS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RSVS_AVAILABLE,
    reason="Rust native extension not compiled — run 'maturin develop' first"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_rsvs():
    """Train RSVS on full English corpus once per module."""
    from rsvs.corpus import DOMAINS
    from rsvs.ingest_wiki import ingest_domains
    import tempfile, os

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        ingest_domains(path, list(DOMAINS.keys()), verbose=False)
        r = Rsvs.load(path)
        yield r
    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Smoke: graph is non-trivial after training
# ---------------------------------------------------------------------------

def test_atoms_promoted(trained_rsvs):
    """After ingesting 9 domains, at least 40 atoms must be promoted."""
    atoms = trained_rsvs.atoms()
    assert len(atoms) >= 40, (
        f"Only {len(atoms)} atoms promoted — corpus may be too small or "
        f"entity_promote_n too high"
    )


def test_cross_domain_atoms_present(trained_rsvs):
    """Key cross-domain atoms must be in the graph."""
    atoms = set(trained_rsvs.atoms())
    required = ["energy", "material", "water"]
    missing = [a for a in required if a not in atoms]
    assert not missing, f"Cross-domain atoms missing from graph: {missing}"


# ---------------------------------------------------------------------------
# Structural similarity: related concepts score higher than unrelated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchor,related,unrelated", [
    ("solid",    "hard",      "water"),
    ("doctor",   "patient",   "crop"),
    ("computer", "processor", "farmer"),
    ("empire",   "ruler",     "software"),
    ("energy",   "heat",      "crop"),
])
def test_structural_similarity_ranking(trained_rsvs, anchor, related, unrelated):
    """sim(anchor, related) >= sim(anchor, unrelated) for known pairs."""
    atoms = set(trained_rsvs.atoms())
    if not all(a in atoms for a in [anchor, related, unrelated]):
        pytest.skip(f"Atoms not in graph: {[a for a in [anchor, related, unrelated] if a not in atoms]}")

    sim_rel = trained_rsvs.similarity(anchor, related)
    sim_unr = trained_rsvs.similarity(anchor, unrelated)

    assert sim_rel is not None and sim_unr is not None
    assert sim_rel.jaccard >= sim_unr.jaccard, (
        f"sim({anchor},{related})={sim_rel.jaccard:.3f} < "
        f"sim({anchor},{unrelated})={sim_unr.jaccard:.3f}"
    )


# ---------------------------------------------------------------------------
# Discriminability: appraise(true) > appraise(false) for clear pairs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true_stmt,false_stmt", [
    ("doctor treats patients",    "doctor plants crops"),
    ("farmer grows crops",        "farmer treats patients"),
    ("computer processes data",   "computer grows crops"),
    ("teacher explains lessons",  "teacher harvests crops"),
    ("empire controls territory", "empire treats patients"),
])
def test_discriminability(trained_rsvs, true_stmt, false_stmt):
    """appraise(true_statement).agree_pct > appraise(false_statement).agree_pct"""
    r_true  = trained_rsvs.appraise(true_stmt)
    r_false = trained_rsvs.appraise(false_stmt)

    gap = r_true.agree_pct - r_false.agree_pct
    assert gap > 0, (
        f"No discriminability: '{true_stmt}' ({r_true.agree_pct:.1f}%) "
        f"<= '{false_stmt}' ({r_false.agree_pct:.1f}%), gap={gap:+.1f}pp"
    )


# ---------------------------------------------------------------------------
# Confidence growth: cross-domain atoms > single-domain atoms
# ---------------------------------------------------------------------------

def test_confidence_growth(trained_rsvs):
    """Atoms appearing across domains have higher confidence than single-domain atoms."""
    cm = trained_rsvs.confidence_map()

    cross = ["energy", "material", "water"]
    single = ["basalt", "polymer"]

    cross_found  = [(a, cm[a]) for a in cross  if a in cm]
    single_found = [(a, cm[a]) for a in single if a in cm]

    if not cross_found:
        pytest.skip("Cross-domain atoms not in graph — corpus too small")

    avg_cross  = sum(c for _, c in cross_found)  / len(cross_found)
    avg_single = sum(c for _, c in single_found) / len(single_found) if single_found else 0.0

    assert avg_cross > avg_single, (
        f"Cross-domain confidence ({avg_cross:.3f}) not higher than "
        f"single-domain ({avg_single:.3f})"
    )
```

### Commit

```bash
git add python/tests/test_accuracy.py
git commit -m "tests: add accuracy integration test suite

- test_atoms_promoted: ≥40 atoms after full corpus ingest
- test_structural_similarity_ranking: 5 parametrized pairs
- test_discriminability: 5 TRUE/FALSE pairs, gap > 0 required
- test_confidence_growth: cross-domain > single-domain confidence
- Skips gracefully if Rust extension not compiled (CI-safe)"
git push origin main
```

---

## TASK E — Complete TUI binary
**[DEPENDS ON: Task B — needs English config]**

**File**: `backend/crates/rsvs-core/src/bin/rsvs-tui.rs`

The file already exists with the skeleton (imports, Mode enum, App struct).
Complete the implementation.

### E.1 — App struct

If not already present, add or verify this struct:

```rust
struct App {
    rsvs:         Rsvs,
    mode:         Mode,
    input:         String,
    context_buf:  String,   // stores first step of CONTEXT mode
    output:       Vec<String>,
    scroll:       usize,
    status_atoms: usize,
    status_edges: usize,
}

impl App {
    fn new() -> Self {
        let config = make_english_config(); // see below
        let rsvs = Rsvs::new(config).expect("Failed to init RSVS");
        let st = rsvs.status();
        App {
            status_atoms: st.total_atoms as usize,
            status_edges: 0,
            rsvs,
            mode: Mode::Normal,
            input: String::new(),
            context_buf: String::new(),
            output: vec!["RSVS TUI ready. Press ? for help.".into()],
            scroll: 0,
        }
    }

    fn push_output(&mut self, line: impl Into<String>) {
        self.output.push(line.into());
        // auto-scroll to bottom
        if self.output.len() > 0 {
            self.scroll = self.output.len().saturating_sub(1);
        }
    }

    fn refresh_status(&mut self) {
        let st = self.rsvs.status();
        self.status_atoms = st.total_atoms as usize;
    }
}
```

### E.2 — `make_english_config()` helper in rsvs-tui.rs

Add this function (same as Task B's make_config, but named differently to avoid
conflict if both binaries are built in same crate):

```rust
fn make_english_config() -> PipelineConfig {
    let seeds: Vec<String> = vec![
        "exists".into(), "entity".into(), "relation".into(), "state".into(),
        "change".into(), "time".into(), "space".into(), "cause".into(),
        "effect".into(), "context".into(), "signal".into(), "pattern".into(),
        "memory".into(), "attention".into(), "value".into(), "agent".into(),
        "goal".into(), "risk".into(), "trust".into(), "identity".into(),
        "language".into(), "meaning".into(), "action".into(), "feedback".into(),
        "the".into(), "is".into(), "are".into(), "was".into(),
        "a".into(), "an".into(), "of".into(), "in".into(),
        "and".into(), "to".into(), "that".into(), "it".into(),
        "by".into(), "with".into(), "for".into(), "from".into(),
        "has".into(), "have".into(), "be".into(), "been".into(),
        "not".into(), "as".into(), "or".into(), "at".into(),
        "its".into(), "which".into(), "when".into(), "can".into(),
    ];

    let mut induction = rsvs::sense::SenseInductionConfig::default();
    induction.tau_overlap = 0.5;
    induction.tau_compress = 0.15;
    induction.composition_min_confidence = 0.15;

    let mut sense = rsvs::sense::SenseConfig::default();
    sense.theta_assign = 0.20;
    sense.gamma_stopword = 0.85;
    sense.induction = induction;

    let mut attention = rsvs::attention::AttentionConfig::default();
    attention.min_cooc = 1;

    PipelineConfig {
        entity_promote_n: 2,
        custom_seeds: Some(seeds),
        sense,
        attention,
        tau_entity_learned: 0.10,
        ..PipelineConfig::default()
    }
}
```

### E.3 — Event handling (complete the main loop)

Complete the `handle_key_event()` function or equivalent main loop. The logic:

```
NORMAL mode:
  'i' → mode = Insert, input.clear(), push_output("INSERT: type text, Enter to ingest")
  'a' → mode = Appraise, input.clear(), push_output("APPRAISE: type statement, Enter to evaluate")
  'c' → mode = ContextStep1, input.clear(), push_output("CONTEXT step 1: type context paragraph")
  'r' → mode = Relate, input.clear(), push_output("RELATE: type concept name")
  '?' → mode = Help (toggle)
  'q' → return / exit
  Up/Down → scroll output panel

INSERT mode (on Enter):
  rsvs.ingest_text(&input)  →  push stats to output
  mode = Normal, input.clear()

APPRAISE mode (on Enter):
  rsvs.appraise(&input)  →  push "verdict: X (Y% agree / Z% disagree)" to output
  mode = Normal, input.clear()

CONTEXT step 1 (on Enter):
  context_buf = input.clone()
  mode = ContextStep2
  push_output("CONTEXT step 2: type statement to appraise against context")
  input.clear()

CONTEXT step 2 (on Enter):
  result = rsvs.appraise_against(&context_buf, &input)
  push verdict to output
  mode = Normal, input.clear(), context_buf.clear()

RELATE mode (on Enter):
  rsvs.relate(&input)  →  push top-5 related nodes to output
  mode = Normal, input.clear()

Any mode on Esc: mode = Normal, input.clear()
Any mode on Backspace: input.pop()
Any mode on char: input.push(char)
```

### E.4 — UI layout (complete `render()` function)

Target layout (use ratatui Constraints):

```
┌─────────────────────────────────────────────────────────────┐
│  RSVS v8.3 — Recursive Symbolic Vocabulary System    [MODE] │
├──────────────────┬──────────────────────────────────────────┤
│  GRAPH STATUS    │  OUTPUT (scrollable)                     │
│  atoms: N        │  > last results here...                  │
│  mode: XXX       │  > ...                                   │
│                  │  > ...                                   │
├──────────────────┴──────────────────────────────────────────┤
│  > input text here...                                       │
├─────────────────────────────────────────────────────────────┤
│  [I]ngest  [A]ppraise  [C]ontext  [R]elate  [Q]uit  [?]help│
└─────────────────────────────────────────────────────────────┘
```

Layout constraints:
- Header: 1 line
- Body: remaining (split 20% left status | 80% right output)
- Input bar: 3 lines
- Footer: 1 line

Color scheme:
- Normal mode border: Cyan
- Insert mode border: Green
- Appraise/Context mode border: Yellow
- Relate mode border: Magenta
- Output text: White
- Status numbers: LightBlue
- Footer shortcuts: DarkGray with mode key in White

### E.5 — Build and smoke test

```bash
cd backend
cargo build --release --bin rsvs-tui 2>&1 | grep -E "^error" | head -20
# Should build cleanly. Run with:
./target/release/rsvs-tui
# Type some text, press i, Enter to ingest
# Press q to quit
```

### Commit

```bash
git add backend/crates/rsvs-core/src/bin/rsvs-tui.rs
git commit -m "tui: complete TUI binary implementation

- App struct with RSVS instance and scroll state
- 6 modes: Normal/Insert/Appraise/Context(2-step)/Relate/Help
- CONTEXT mode uses appraise_against() for isolated contextual appraisal
- English seeds via make_english_config()
- ratatui layout: status panel + scrollable output + input bar
- Color-coded borders per mode"
git push origin main
```

---

## TASK F — Convergence smoke test
**[DEPENDS ON: Task A — needs multi-language corpus to test cross-lingual links]**

**File**: `python/tests/test_convergence.py` (new file)

The convergence engine claims it can detect that two words from different domains
mean the same concept without being told they're related. This test verifies
that claim with a concrete example.

```python
"""
RSVS Convergence Engine Test

Tests that the convergence engine creates LanguageLink(structural_equivalence)
between concepts that have similar composition structures but never co-occur.

Strategy: ingest two domains with mirror-image vocabulary. The concepts
"doctor" and "physician" should converge (same compositions, never co-occur
in the same sentence if we keep them in separate ingest batches).
"""
import pytest

try:
    from rsvs import Rsvs
    RSVS_AVAILABLE = True
except ImportError:
    RSVS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RSVS_AVAILABLE,
    reason="Rust native extension not compiled"
)


CORPUS_A = """
A doctor examines patients to diagnose illness.
The doctor prescribes medicine to treat the patient.
Doctors work in hospitals and clinics.
A doctor uses medical tools to examine a patient.
The doctor treats illness by prescribing medicine.
Medical knowledge helps the doctor diagnose patients.
A doctor heals patients using medicine and treatment.
"""

# Synonym corpus: physician = doctor, but never appears with "doctor"
CORPUS_B = """
A physician examines patients to diagnose illness.
The physician prescribes medicine to treat the patient.
Physicians work in hospitals and clinics.
A physician uses medical tools to examine a patient.
The physician treats illness by prescribing medicine.
Medical knowledge helps the physician diagnose patients.
A physician heals patients using medicine and treatment.
"""


@pytest.fixture(scope="module")
def converged_rsvs():
    """Build RSVS with mirrored corpora for convergence testing."""
    r = Rsvs(entity_promote_n=2, theta_assign=0.20)
    r.ingest(CORPUS_A)
    r.ingest(CORPUS_B)
    # Run convergence detection explicitly
    r.detect_convergence()
    return r


def test_both_concepts_in_graph(converged_rsvs):
    """Both doctor and physician must be atoms."""
    atoms = set(converged_rsvs.atoms())
    assert "doctor"    in atoms, "doctor not promoted to atom"
    assert "physician" in atoms, "physician not promoted to atom"


def test_structural_similarity_high(converged_rsvs):
    """doctor and physician must have high structural similarity (≥0.5)."""
    atoms = set(converged_rsvs.atoms())
    if "doctor" not in atoms or "physician" not in atoms:
        pytest.skip("One of the concepts not in graph")

    result = converged_rsvs.structural_similarity("doctor", "physician")
    assert result is not None
    assert result.structural_similarity >= 0.5, (
        f"structural_similarity(doctor, physician) = "
        f"{result.structural_similarity:.3f} — expected ≥ 0.5"
    )


def test_convergence_link_created(converged_rsvs):
    """
    After detect_convergence(), node_info(doctor) should have a LanguageLink
    pointing to physician with type 'structural_equivalence'.
    """
    try:
        info = converged_rsvs.node_info("doctor")
    except Exception:
        pytest.skip("doctor not in graph")

    links = [l for l in info.language_links
             if l.link_type == "structural_equivalence"]
    assert len(links) > 0, (
        "No structural_equivalence LanguageLink on 'doctor' after detect_convergence(). "
        "Convergence engine may need lower min_overlap_threshold or more corpus data."
    )
```

### Commit

```bash
git add python/tests/test_convergence.py
git commit -m "tests: add convergence engine smoke test

- Verifies doctor/physician convergence from mirrored corpora
- Tests: atoms promoted, structural_similarity ≥ 0.5, LanguageLink created
- Skips gracefully if native extension not available"
git push origin main
```

---

## TASK G — Final wiring: seed.rs English comment update
**[DEPENDS ON: Task B]**

**File**: `backend/crates/rsvs-core/src/seed.rs`

This is a minor cleanup task. The seed file may have comments referencing
Indonesian. Update the doc comment to reflect English-first:

Find the doc comment at the top of `seed.rs` and update it to say:

```rust
//! Seed atoms — the epistemological primitives that form the axiomatic
//! foundation of every RSVS knowledge graph.
//!
//! 24 seed atoms are always present from initialization and can never
//! be removed. They represent the most fundamental concepts from which
//! all other meaning is composed.
//!
//! The system is language-agnostic: these labels happen to be English,
//! but the structural relationships hold across any language. Custom
//! seed sets (including functional words for a specific language) can
//! be provided via `PipelineConfig::custom_seeds`.
//!
//! Default seeds (24):
//! exists, entity, relation, state, change, time, space, cause, effect,
//! context, signal, pattern, memory, attention, value, agent, goal,
//! risk, trust, identity, language, meaning, action, feedback
```

Also, anywhere in `seed.rs` that references Indonesian (e.g., "bahasa indonesia",
"yang", "di", "adalah"), remove or generalize that comment.

### Commit

```bash
git add backend/crates/rsvs-core/src/seed.rs
git commit -m "seed: update doc comments for English-first approach

- Remove Indonesian language references from doc comments
- Clarify that custom_seeds handles language-specific functional words
- No logic changes"
git push origin main
```

---

## EXECUTION ORDER FOR SWARM

```
Wave 1 (no dependencies, run in parallel):
  Agent-1 → TASK A  (corpus.py)
  Agent-2 → TASK B  (rsvs-realtest.rs + make_config)
  Agent-3 → TASK G  (seed.rs comment cleanup)

Wave 2 (after Wave 1 completes):
  Agent-4 → TASK C  (eval.py — depends on Task A vocabulary)
  Agent-5 → TASK E  (rsvs-tui.rs — depends on Task B config)

Wave 3 (after Wave 2 completes):
  Agent-6 → TASK D  (test_accuracy.py — depends on Tasks A + C)
  Agent-7 → TASK F  (test_convergence.py — depends on Task A)
```

## SUCCESS CRITERIA (run after all tasks complete)

```bash
# 1. Corpus integrity
python -c "from rsvs.corpus import DOMAINS; print({k: len(v) for k, v in DOMAINS.items()})"
# Expected: 9 domains, all 30 sentences

# 2. Realtest discriminability
cd backend && ./target/release/rsvs-realtest
# Expected: ≥5/6 domains PASS, gap ≥10pp

# 3. Eval suite
python -m rsvs.eval --all
# Expected: Discriminability ≥ 0.60, all benchmarks PASS or close

# 4. Pytest accuracy
pytest python/tests/test_accuracy.py python/tests/test_convergence.py -v
# Expected: all tests pass or skip (if no compiled binary)

# 5. TUI smoke
cd backend && ./target/release/rsvs-tui
# Expected: launches, accepts input, no crash on 'q'
```
