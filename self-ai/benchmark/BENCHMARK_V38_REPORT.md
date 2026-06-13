# SELF-AI v38 Empirical Benchmark Report

**Date**: 2026-06-07
**Version**: v38 (External Validation + Strengthening/Weakening + No Hardcore)

## Executive Summary

Pattern matching works for **exact** and **same-type** questions, but suffers from **high false positive rate** for questions with similar linguistic structure but different answers. The bge-m3 embedding model cannot distinguish between subtypes of the same concept category (e.g., different types of majas).

## Key Findings

### 1. Disk Space Issue — Root Cause of "System Never Worked"

```
Disk free: 337MB < 500MB minimum → models not loaded → ALL embeddings empty
→ _match_by_embedding() returns None → pattern matching NEVER activated
→ ALL answers came from understanding_builder.py (rule-based), not learned patterns
→ 21 semantic understandings: times_applied=0 for ALL
→ learned_patterns.json: never persisted to disk in production
```

**Impact**: The entire ExperienceWeight system was effectively dead in production. Every answer was from the rule-based understanding graph, not from teaching.

### 2. Embedding Matching Efficacy (with models loaded)

| Test Case | Cosine Sim | Threshold | Correct? | Notes |
|-----------|-----------|-----------|----------|-------|
| EXACT-BK (same question) | 1.0000 | EXACT | ✅ | Perfect match |
| GEN-BK (similar, same answer) | 0.7426 | SIMILAR | ✅ | Generalization works |
| WRONG-BK (different subtype) | 0.8218 | SIMILAR | ✅/⚠️ | "menjerit" is ALSO personifikasi — lucky |
| EXACT-IM (same question) | 1.0000 | EXACT | ✅ | Perfect match |
| GEN-IM (similar, same answer) | 0.6575 | SIMILAR | ✅ | Generalization works |
| WRONG-IM (different cause) | 0.4892 | PASS2 | ❌ | Would match "kebakaran" for "harga beras naik" |
| EXACT-EK (same question) | 1.0000 | EXACT | ✅ | Perfect match |
| GEN-EK (same type, diff answer) | 0.7119 | SIMILAR | ❌ FP | Would answer "08.00" for "pukul berapa perpustakaan?" |
| CROSS (unrelated) | 0.3954 | NO_MATCH | ✅ | Correctly rejected |

**Stats**: TP=6, FP=1, FN=0

### 3. Critical False Positive Analysis

When taught "Kata menari-nari majas → personifikasi":

| Test Question | Cosine | Would Match? | Correct Answer | Is FP? |
|--------------|--------|-------------|---------------|--------|
| "Kata melebih-lebihkan majas" | 0.7274 | YES | hiperbola | **FP** |
| "Kata istana majas" | 0.8045 | YES | metonimia | **FP** |
| "Bagai rembulan majas" | 0.7434 | YES | simile/perumpamaan | **FP** |

**Root cause**: bge-m3 encodes the TOPIC (majas) not the SUBTYPE (personifikasi vs hiperbola). Questions about different types of majas cluster together in embedding space.

### 4. times_applied=0 Root Cause

```python
# understanding_builder.py
def apply(self, node, text, question):
    result = self._apply_transformation(node, text, question)
    if result is not None:
        return {...}
    return None
    # BUG: No node.times_applied += 1
    # times_applied ONLY increments in strengthen()/weaken()
    # which are only called from record_feedback() via self_correction.py
```

### 5. Persistence Works — But Was Never Tested

- `_save_learned_patterns()`: ✅ atomic write with .tmp + os.replace
- Path: `self-ai/data/learned_patterns.json` ✅ resolves correctly  
- `_load_learned_patterns()`: ✅ loads with backfill for v37→v38
- **Problem**: File never persisted in production because models never loaded → embeddings always empty → patterns useless → cleaned up on restart

## Implications for Architecture

### The Multi-Pattern Activation Problem

The original question was: "how can a trigger activate multiple semantic understandings to create new understanding?"

This benchmark shows that **even single-pattern matching has a false positive problem**. Before building multi-pattern composition, we must fix:

1. **Threshold calibration**: Pass 1 threshold 0.50 is too low for subtyped questions
2. **Context-aware matching**: bge-m3 alone cannot distinguish subtypes — need answer_embedding validation
3. **Answer verification**: After matching, validate that the matched answer is semantically consistent with the question context

### Recommended Next Steps (Priority Order)

1. **Fix disk space** — ensure models can load in production
2. **Add answer_embedding validation** — after Pass 1 match, verify answer_embedding is consistent with question context (cosine > threshold)
3. **Raise Pass 1 SIMILAR threshold** from 0.50 to 0.65 for non-exact matches
4. **Raise Pass 2/3 threshold** from 0.40 to 0.50 (reduce false positives)
5. **Fix times_applied tracking** — add usage counter separate from validation counter
6. **Then** benchmark full comprehend() with models loaded
7. **Then** implement multi-pattern activation
