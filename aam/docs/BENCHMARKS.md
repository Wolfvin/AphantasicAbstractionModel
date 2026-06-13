# Benchmark Documentation

> Performance characteristics and benchmarking guide for RSVS (Recursive Symbolic Vocabulary System)

---

## Table of Contents

1. [Overview of Criterion Benchmarks](#1-overview-of-criterion-benchmarks)
2. [How to Run Benchmarks](#2-how-to-run-benchmarks)
3. [Interpreting Results](#3-interpreting-results)
4. [Known Performance Characteristics](#4-known-performance-characteristics)
5. [Tips for Optimization](#5-tips-for-optimization)

---

## 1. Overview of Criterion Benchmarks

RSVS uses [Criterion.rs](https://bheisler.github.io/criterion.rs/book/criterion_rs.html) for statistical benchmarking with rigorous confidence intervals. All benchmarks live in `layer1/crates/rsvs-core/benches/rsvs_bench.rs`.

### Benchmark Suite

| Benchmark | What It Measures | Typical Time |
|-----------|-----------------|--------------|
| `jaccard_100_elements` | Jaccard similarity between two 100-element atom sets | ~2 µs |
| `npmi_lookup` | Single NPMI table lookup from co-occurrence stats | ~50 ns |
| `cooc_ingest_sentence_20_tokens` | Co-occurrence stats ingestion for a 20-token sentence | ~5 µs |
| `sense_ingest_10_atoms` | Sense assignment for a 10-atom context | ~15 µs |
| `pipeline_ingest_text` | Full pipeline: tokenize → attention → sense → autonomy → persist | ~800 µs |

### Architecture Mapping

```
pipeline_ingest_text (~800 µs)
├── tokenize + split_sentences    (~10 µs)
├── cooc_ingest_sentence_20_tokens × N sentences
├── npmi_lookup × M token pairs
├── jaccard_100_elements × K nodes
├── sense_ingest_10_atoms × K nodes
├── autonomy.update_confidence     (~1 µs per node)
└── persist + events               (~50 µs)
```

---

## 2. How to Run Benchmarks

### Prerequisites

- Rust toolchain (1.75+)
- The `rsvs-core` crate builds without the `python` feature

### Running All Benchmarks

```bash
cd layer1
cargo bench
```

### Running a Specific Benchmark

```bash
cd layer1
cargo bench --bench rsvs_bench -- jaccard
cargo bench --bench rsvs_bench -- npmi
cargo bench --bench rsvs_bench -- cooc
cargo bench --bench rsvs_bench -- sense
cargo bench --bench rsvs_bench -- pipeline
```

### With Verbose Output

```bash
cargo bench -- --verbose
```

### Saving Baseline for Comparison

```bash
# Save current performance as baseline
cargo bench -- --save-baseline main

# After changes, compare against baseline
cargo bench -- --baseline main
```

### CI Integration

Benchmarks run automatically in CI (`.github/workflows/ci.yml`):

```yaml
benchmark:
  name: Rust – benchmarks
  runs-on: ubuntu-latest
  steps:
    - name: Run benchmarks
      run: cargo bench --workspace -- --save-baseline ci-baseline

    - name: Store benchmark results
      uses: actions/upload-artifact@v4
      with:
        name: criterion-benchmarks
        path: layer1/target/criterion/
        retention-days: 30
```

---

## 3. Interpreting Results

### Criterion Output

After running `cargo bench`, Criterion generates:

1. **Console output**: Mean, median, standard deviation, and confidence intervals
2. **HTML reports**: `target/criterion/<benchmark>/report/index.html`
3. **Comparison reports**: When a baseline exists, shows regression/improvement

### Example Console Output

```
jaccard_100_elements
                    time:   [1.9843 µs 2.0121 µs 2.0433 µs]
                    change: [-2.3456% -1.1234% +0.0987%] (p = 0.08 > 0.05)
                    No change in performance detected.
```

- **time**: Lower bound, estimate, upper bound (95% confidence)
- **change**: Performance change relative to baseline. Negative = faster.
- **p-value**: If p < 0.05, the change is statistically significant.

### Understanding Regressions

| Indicator | Meaning |
|-----------|---------|
| 🟢 `No change` | Performance within noise threshold |
| 🟡 `Regressed` | Performance degraded (slower). Check if acceptable. |
| 🔴 `Regressed (significant)` | Large performance regression. Investigate immediately. |
| 🔵 `Improved` | Performance improved (faster). |

### HTML Reports

Open `target/criterion/<benchmark>/report/index.html` in a browser for:

- **Kernel density plot**: Distribution of measured times
- **Regression plot**: Performance over iterations
- **Comparison chart**: Before vs. after (when baseline exists)

---

## 4. Known Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Jaccard similarity | O(|A| + |B|) | Linear in atom set sizes |
| NPMI lookup | O(1) | HashMap lookup |
| Co-occurrence ingest | O(n²) | n = tokens per sentence (typically 5–20) |
| Sense ingest | O(S × K) | S = number of senses, K = core size |
| Full pipeline ingest | O(T × S × K) | T = tokens, S = senses, K = core atoms |
| Relate (Jaccard) | O(N × |A|) | N = total nodes, parallelized via rayon |

### Memory Usage

| Component | Approximate Memory | Scaling |
|-----------|-------------------|---------|
| Graph (nodes + edges) | ~100 bytes/node | Linear with graph size |
| Co-occurrence stats | ~50 bytes/pair | Quadratic with vocabulary (bounded by sentence length) |
| Sense managers | ~200 bytes/sense | Linear with sense count |
| Entity detector | ~20 bytes/token | Linear with vocabulary |
| Event buffer | ~100 bytes/event | Bounded by consumption frequency |

### Performance Hotspots

1. **Sense assignment** (`sense_ingest`): Candidate pruning mitigates the O(S × K) complexity. The `ceil(ln(|senses| + 1))` threshold limits scoring to relevant senses only.

2. **Jaccard computation** (`jaccard_100_elements`): Parallelized via `rayon` for `relate()` mode. Each node's Jaccard is computed independently.

3. **Co-occurrence pairs** (`cooc_ingest_sentence_20_tokens`): O(n²) per sentence, but n is typically small (5–20 tokens after filtering). For large documents, sentence-level batching keeps memory bounded.

4. **Confidence EMA update**: O(1) per node. Global stability check is O(N) but only triggers on large batch deltas.

### Cold Start vs. Warm Performance

- **Cold start** (empty graph): First ingest creates seed nodes and builds initial co-occurrence tables. Expect ~2× slower than steady state.
- **Warm state** (existing graph): Ingest benefits from pre-built co-occurrence tables and existing sense structures.

---

## 5. Tips for Optimization

### 1. Profile Before Optimizing

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph for the pipeline benchmark
cargo flamegraph --bench rsvs_bench -- --pipeline
```

### 2. Use Rayon for Parallelism

The `relate()` mode already uses `rayon::par_iter()` for Jaccard computation. When adding new hot loops:

```rust
use rayon::prelude::*;

// Parallel iteration over nodes
let results: Vec<_> = node_ids
    .par_iter()
    .filter_map(|&id| compute_score(id))
    .collect();
```

### 3. Reduce Allocations

Criterion benchmarks help identify allocation-heavy code:

- Use `Vec::with_capacity()` when sizes are known
- Reuse buffers across iterations (e.g., `CoocStats` is reused)
- Consider `SmallVec` for small atom sets (most nodes have < 10 atoms)

### 4. Optimize Sense Assignment

The sense assignment threshold (`θ_assign`) controls how many senses are scored per ingest:

- **Higher θ_assign** → fewer senses scored → faster but less accurate
- **Lower θ_assign** → more senses scored → slower but more accurate

The adaptive threshold (`mean(history) + k1·std(history)`) auto-tunes this.

### 5. Batch Ingest for Throughput

For bulk data loading, batch sentences into larger chunks:

```python
# Instead of many small ingests
for sentence in sentences:
    bridge.run(mode="ingest", text=sentence)

# Batch into larger texts
batch = ". ".join(sentences[:100])
bridge.run(mode="ingest", text=batch)
```

This amortizes the pipeline overhead across more tokens.

### 6. Monitor Memory Growth

If memory grows unexpectedly:

- Check `status.flip_count` for quarantined nodes (they accumulate but don't process)
- Check sense count — fragile senses should be pruned after `k_fragile` (30) inactivity
- Verify `rsvs-state.json` isn't growing unbounded (persist + load roundtrip)

### 7. Use Release Builds

Always benchmark with `--release`:

```bash
cargo bench  # Automatically uses release profile

# For manual testing:
cargo build --release
./target/release/rsvs-smoke
```

The `Cargo.toml` has `codegen-units = 1` in `[profile.release]` for maximum optimization.
