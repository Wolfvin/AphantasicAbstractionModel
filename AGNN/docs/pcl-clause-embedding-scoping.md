# PCL Clause Embedding — Scoping Document

**Date:** 2026-06-22
**Mode:** **Scoping only — no code changes.** Per BOS/researcher review after sprint round 10 (two reverted attempts: `final_bucket_anchors`, `belalai` index-0 morphology guard), this document exists so the next attempt has a design to implement against instead of another ad-hoc patch.

---

## 0. TL;DR

PCL currently treats every sentence as a **flat token sequence**: one AGENT span, one ACTION token, one OBJECT span, full stop. Round 9 (predicate-final adjectives) and round 5 (clause coordinators) both extended this flat model successfully. Round 10 hit a DIFFERENT kind of gap: sentences where a whole clause (`"harga naik"` — *price rises*) fills the AGENT or OBJECT slot of an outer clause (`"...menyebabkan harga naik"` — *...causes price [to] rise*). This is **not** a missing bucket signal — it's a missing **structural concept** (a clause nested inside another clause's argument slot). No bucket-anchor generalisation fixes this; the flat-sequence assumption itself needs to grow one more dimension.

This is corpus-confirmed as common, not a rare edge case: the causal-relation corpus (`pretrain_corpus.txt`, the single largest corpus file) is built almost entirely around the pattern `"X menyebabkan/memicu/mengakibatkan [Y berubah]"`, where `[Y berubah]` is a sentential object. PCL currently "works" on this corpus only because `_extract_action_object` blindly grabs the LAST token as the object — `"harga naik"` degrades to object=`"naik"`, silently losing `"harga"`. That is the un-investigated reason round 10's bare-root-verb tokens (`naik`, `turun`, `tumbuh`, `muncul`) have ambiguous positional signatures: they aren't ambiguous verbs, they're embedded predicates whose outer-clause role (subject-filler vs object-filler) determines which flat bucket they land in.

---

## 1. What "clause" means for PCL (Question 1)

PCL already has an implicit clause unit: the span `_parse_clause_spo` operates on — one ACTION (or predicate-final token, since round 9) plus the AGENT/OBJECT tokens around it. Call this a **simple clause**. The existing anchor-split mechanism (rounds 1, 4, 5) already produces MULTIPLE simple clauses from one sentence, but only as **siblings** — coordinate ("X dan Y", round 5) or sequential-with-a-connector (subordinate, round 4). Both are flat: clause A, then clause B, never one inside the other.

**New concept needed: embedded clause.** A simple clause that does not stand on its own at the top level of the sentence, but instead fills the AGENT or OBJECT slot of another (outer) simple clause.

```
[harga naik]         menyebabkan        [protes warga]
^ embedded clause    ^ outer ACTION     ^ outer OBJECT
  (fills outer AGENT slot)
```

```
panas menyebabkan    [suhu naik]
^ outer AGENT ^ outer ACTION
                     ^ embedded clause (fills outer OBJECT slot)
```

Definition for PCL's purposes: **an embedded clause is a simple-clause-shaped sub-span (its own AGENT+ACTION, with no OBJECT of its own in this corpus's patterns) found entirely within what would otherwise be the AGENT or OBJECT span of an outer clause.** This is deliberately narrower than full linguistic complementation (no relative clauses, no "yang"-clauses here — those are already handled adequately by round 7's finding that `tag_sentence` generalises to them without special-casing). Scope is limited to the **sentential-subject / sentential-object** pattern the corpus evidence above shows is actually load-bearing.

---

## 2. Representation that stays zero-gradient (Question 2)

### 2.1 Detection signal — reuse, don't invent

Detection must come from signals PCL already computes, the same discipline as every round so far:

- `_find_action_positions(tokens)` already returns ALL recognised action positions in a clause, in order. Today, when there are 2+ action positions in one (post-anchor-split) clause, `_parse_clause_spo` either defensively `continue`/`break`s on the second one (silently dropping it) or — for `spo_all()`'s purposes — treats the SAME boundary mechanism (`_detect_clause_anchors`) as the only legitimate way to split.
- **New, purely structural test, zero new corpus signal:** when a sub-clause (after the EXISTING anchor-split has already run and found no coordinator/subordinator boundary) still contains 2 action positions `i < j`, classify the relationship as embedding (not a parse error) when:
  - `j` is the LAST action position and sits in what would be the OBJECT span of `i` (`i`'s clause has no anchor-split boundary between them) → **sentential object**: outer predicate = token at `i`, embedded predicate = token at `j`, embedded AGENT = tokens between `i` and `j` that aren't particles.
  - `i` is the FIRST action position and sits in what would be the AGENT span of `j` (i.e. `i` precedes the OUTER predicate `j` with no recognised AGENT before `i`) → **sentential subject**: embedded predicate = token at `i`, embedded AGENT = tokens before `i`, outer predicate = token at `j`.
  - This is symmetric with round 1's post-verbal-particle exclusion and round 5's coordinator detection: both already use "is there ANOTHER recognised action adjacent to this one" as a pure positional/structural fact, not a new word list.

### 2.2 Data structure — nested, not flat

`SPO` (the existing dataclass: `subject: str, predicate: str, object: str, raw: str, negated: bool`) cannot represent this without a breaking change to its field types. Two options, evaluated:

| Option | Description | Verdict |
|---|---|---|
| A. Change `SPO.subject`/`.object` to `Union[str, SPO]` | Recursive, accurate, breaks every existing caller that assumes `str` | Rejected — round 6 (`spo_all`) was explicitly designed as an ADDITIVE extension specifically to avoid breaking `spo()`'s contract. Changing field types now would violate that precedent and break `classify()`, which does `self._normalize_token(spo.predicate)` assuming a string. |
| B. New method `spo_embedded(text) -> EmbeddedSPO`, additive | New dataclass: `EmbeddedSPO(outer: SPO, embedded: SPO, embedded_role: Literal["subject", "object"])`. `spo()`/`spo_all()` unchanged; callers that don't care about embedding never see it. | **Recommended** — same additive pattern as round 6's `spo_all()`. `_parse_clause_spo`'s existing defensive "second action → continue/break" becomes the trigger: instead of silently dropping, return enough information for `spo_embedded()` to build both halves by calling `_parse_clause_spo` recursively on the two sub-spans. |

No `nn.Parameter`, no learned weights, no new training pass — this is a structural parse-time decision built entirely from `_find_action_positions`' existing output, same as everything else in PCL.

### 2.3 Boundary with existing anchor-split

Critical ordering constraint: embedding detection MUST run strictly **after** the existing coordinate/subordinate anchor-split (rounds 1/4/5) has already partitioned the sentence into sibling clauses, and must only look for a SECOND action **within** one already-isolated sibling clause. Skipping this ordering risks exactly the round-4.5 failure mode (two structural heuristics stepping on each other) — coordinated clauses ("X dan Y") must never be mistaken for embedding.

---

## 3. Minimal test cases before this counts as "working" (Question 3)

All four must pass without corpus changes (all four sentence shapes already exist verbatim or near-verbatim in `pretrain_corpus.txt`):

1. **Sentential object** — `"panas menyebabkan suhu naik"` → `spo_embedded()` returns outer=(subject="panas", predicate="menyebabkan"), embedded=(subject="suhu", predicate="naik"), embedded_role="object". Today: `spo().object == "naik"` (silently drops "suhu").
2. **Sentential subject** — `"harga naik menyebabkan protes warga"` → outer=(predicate="menyebabkan", object="protes warga"), embedded=(subject="harga", predicate="naik"), embedded_role="subject". Today: `spo().subject` swallows "harga naik menyebabkan" as one run-on AGENT span (confirmed broken in the round-9 audit's Wikipedia test, structurally identical sentence shape).
3. **Non-embedded regression guard** — `"kucing makan ikan"` and every existing round 1-9 regression test must be COMPLETELY unaffected. `spo_embedded()` on a non-embedded sentence should either raise/return `None` or fall back to wrapping the flat `spo()` result — never invent a spurious embedding.
4. **Coordinator/embedding non-interference** — `"ayah membaca koran dan ibu memasak nasi"` (round 5's coordinator test) must still split into 2 SIBLING clauses via the existing mechanism, NOT get mis-read as one clause embedded in the other.

Success bar for closing this as a round: all 4 pass, plus the full existing suite (450 tests, post-round-9) stays green. If achieving (1)+(2) requires touching `_is_action_token`/`_find_action_positions` in a way that changes round 1-9 behavior, that is the round-10 regression pattern repeating and the attempt should be reverted, same discipline as before — not pushed through because "it's only a little different."

---

## 4. Explicitly out of scope

- Relative clauses ("yang") — round 7 found these already generalise adequately without special-casing.
- Multi-level nesting (a clause embedded inside a clause embedded inside another) — corpus has zero evidence of this pattern; do not build for it speculatively.
- Comparative/superlative (`"lebih X dari Y"`, `"paling X"`) — a separate, unrelated gap (round 9/10 notes), not addressed by this design.
- Any change to `SPO`'s existing field types or `spo()`'s return contract.

---

## 5. Open question for BOS before implementation

Round 10's two reverted attempts both looked correct in isolation and only failed under the full test suite / held-out corpus check. Before writing `spo_embedded()`, run the EXACT same discipline: implement behind a NEW additive method only, verify all 4 test cases plus the full 450-test suite, AND re-run the round-9 held-out Wikipedia "Gajah" sentences (not just the synthetic corpus) before considering this mergeable — that held-out check is what caught problems the test suite alone missed twice already this sprint.
